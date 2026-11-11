"""
demo/scripts/run_complete_demo.py
=================================
Reproducible end-to-end DBMedic Phase 6 demonstration script.

Runs the full incident lifecycle cleanly using IncidentOrchestrator:
  1. Environment check
  2. Incident creation & baseline telemetry capture
  3. Diagnosis (explicit mode: deterministic, live_tiny_aya, or simulated)
  4. Evidence validation & remediation preparation
  5. Explicit human operator approval (interactive confirmation required unless
     --operator and --confirm-human-approval are provided by an operator CLI)
  6. Safe allowlisted index execution & before/after benchmark verification
  7. Unified evaluation & consolidated report check

Usage:
    # Primary live PostgreSQL demo (deterministic engine + live dbmedic_lab):
    python demo/scripts/run_complete_demo.py --scenario slow_query_missing_index --mode deterministic

    # Live Tiny Aya + live PostgreSQL demo (requires HF_TOKEN + GPU + dbmedic_lab):
    python demo/scripts/run_complete_demo.py --scenario slow_query_missing_index --mode tiny-aya

    # Offline simulated demo (strictly tagged simulated=true):
    python demo/scripts/run_complete_demo.py --scenario slow_query_missing_index --mode simulated
"""

from __future__ import annotations

import argparse
import getpass
import json
import sys
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env", override=False)

from agent.core.audit_logger import redact_secrets
from agent.core.environment_check import generate_readiness_report
from agent.core.incident_orchestrator import (
    EXECUTION_MODE_SIMULATED,
    IncidentOrchestrator,
    OrchestrationBlockedError,
    normalize_execution_mode,
)
from agent.evaluation.final_evaluator import (
    generate_consolidated_report,
    run_unified_evaluation,
)


def load_scenario(scenario_name: str) -> dict[str, Any]:
    path = PROJECT_ROOT / "demo" / "scenarios" / f"{scenario_name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Demo scenario '{scenario_name}' not found at {path}.")
    return json.loads(path.read_text(encoding="utf-8"))


def run_demo(
    *,
    scenario_name: str = "slow_query_missing_index",
    mode: str = "deterministic",
    operator: str | None = None,
    confirm_human_approval: bool = False,
    n_runs: int = 7,
    incidents_dir: Path | None = None,
    approval_store_path: Path | None = None,
    audit_log_path: Path | None = None,
    comparison_output_path: Path | None = None,
    final_metrics_path: Path | None = None,
    report_output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run complete end-to-end DBMedic demo workflow cleanly."""
    scenario = load_scenario(scenario_name)
    exec_mode = normalize_execution_mode(mode)

    from agent.core.approval_manager import ApprovalManager
    from agent.core.audit_logger import AuditLogger
    from agent.core.remediation_manager import RemediationManager

    audit_logger = AuditLogger(log_path=audit_log_path)
    approval_mgr = ApprovalManager(store_path=approval_store_path, audit_logger=audit_logger)
    rem_mgr = RemediationManager(approval_manager=approval_mgr, audit_logger=audit_logger)
    orchestrator = IncidentOrchestrator(
        incidents_dir=incidents_dir,
        audit_logger=audit_logger,
        approval_manager=approval_mgr,
        remediation_manager=rem_mgr,
        comparison_output_path=comparison_output_path,
    )

    print("=" * 72)
    print(f"DBMedic Complete Demo — Scenario: {scenario['scenario_id']}")
    print(f"Execution Mode: {exec_mode} (simulated={exec_mode == EXECUTION_MODE_SIMULATED})")
    if exec_mode == EXECUTION_MODE_SIMULATED:
        print("[NOTICE] Running in EXPLICIT SIMULATED MODE — outputs are synthetic.")
    print("=" * 72)

    readiness = generate_readiness_report(save=False)
    print(f"[1/7] Readiness status: {readiness['overall_status']} (live_db={readiness['capabilities']['live_postgres_ready']}, live_ai={readiness['capabilities']['live_tiny_aya_ready']})")

    sim_idx = bool((scenario.get("precondition") or {}).get("idx_orders_customer_id_exists", False))
    incident = orchestrator.create_incident(
        incident_id=scenario["incident_id"],
        query_description=scenario["user_prompt"],
        customer_id=scenario["customer_id"],
        execution_mode=exec_mode,
        simulated_index_exists=sim_idx,
    )
    print(f"[2/7] Created incident '{incident['incident_id']}' (run_id={incident['run_id']}, status={incident['status']})")

    incident = orchestrator.collect_incident_evidence(incident, n_benchmark_runs=n_runs)
    pre = incident["pre_remediation_metrics"]
    print(f"[3/7] Collected evidence: scan={pre['scan_type']}, median={pre['median_execution_time_ms']} ms")

    incident = orchestrator.run_agent_diagnosis(incident)
    incident = orchestrator.validate_diagnosis(incident)
    diag = incident["diagnosis"]
    print(f"[4/7] Diagnosis complete: root_cause={diag['root_cause']['type']}, status={incident['status']}")

    if incident["diagnosis_validation"].get("requires_remediation"):
        incident = orchestrator.prepare_remediation(incident)
        incident = orchestrator.request_human_approval(incident)
        req = incident["approval_request"]
        print(f"[5/7] Awaiting human approval for SQL: {req['proposed_sql']};")

        if confirm_human_approval:
            chosen_op = operator or "human_operator"
            confirm_method = "explicit_human_operator"
        else:
            if not sys.stdin.isatty():
                raise PermissionError(
                    "Interactive confirmation required (or supply --confirm-human-approval --operator <name>)."
                )
            chosen_op = operator or (input(f"Operator name [{getpass.getuser()}]: ").strip() or getpass.getuser())
            resp = input("Type 'APPROVE' to authorize DDL execution: ").strip()
            if resp != "APPROVE":
                incident = orchestrator.submit_human_decision(
                    incident,
                    approved=False,
                    operator=chosen_op,
                    reason="Operator declined during demo",
                )
                incident = orchestrator.close_incident(incident)
                return {"incident": incident, "approved": False}
            confirm_method = "interactive_cli"

        incident = orchestrator.submit_human_decision(
            incident,
            approved=True,
            operator=chosen_op,
            confirmation_method=confirm_method,
        )
        print(f"      Approved by '{chosen_op}' -> status={incident['status']}")

        incident = orchestrator.execute_approved_remediation(incident)
        incident = orchestrator.verify_remediation(incident, n_benchmark_runs=n_runs)
        incident = orchestrator.close_incident(incident)

        cmp_rep = incident["verification_report"]
        perf = cmp_rep["performance_comparison"]
        print(
            f"[6/7] Verified remediation: {perf['baseline_scan']} ({perf['baseline_median_ms']} ms) "
            f"-> {perf['after_scan']} ({perf['after_remediation_median_ms']} ms), "
            f"{perf['improvement_pct']}% faster, final_status={incident['status']}"
        )
    else:
        incident = orchestrator.close_incident(incident)
        print(f"[5-6/7] No DDL needed; closed incident cleanly as {incident['status']}")

    final_metrics = run_unified_evaluation(
        output_path=final_metrics_path,
        run_redteam=True,
        audit_logger=audit_logger,
    )
    rep_paths = generate_consolidated_report(output_dir=report_output_dir)
    print(
        f"[7/7] Evaluation complete (mode={final_metrics['evaluation_mode']}, "
        f"real_episodes={final_metrics['episode_counts']['num_real_episodes']}, "
        f"simulated={final_metrics['episode_counts']['num_simulated_episodes']})"
    )
    print(f"      Summary report: {rep_paths['md_path']}")
    return {
        "incident": incident,
        "approved": True,
        "final_metrics": final_metrics,
        "report_paths": rep_paths,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run complete DBMedic Phase 6 demo.")
    parser.add_argument(
        "--scenario",
        default="slow_query_missing_index",
        choices=[
            "slow_query_missing_index",
            "slow_query_existing_index",
            "slow_query_selective_filter",
        ],
    )
    parser.add_argument("--mode", default="deterministic", choices=["deterministic", "tiny-aya", "simulated"])
    parser.add_argument("--operator", default=None, help="Human operator name.")
    parser.add_argument(
        "--confirm-human-approval",
        action="store_true",
        help="Explicit operator confirmation flag for scripted demo runs.",
    )
    parser.add_argument("--runs", type=int, default=7)
    args = parser.parse_args()

    try:
        run_demo(
            scenario_name=args.scenario,
            mode=args.mode,
            operator=args.operator,
            confirm_human_approval=args.confirm_human_approval,
            n_runs=args.runs,
        )
        return 0
    except OrchestrationBlockedError as exc:
        print(f"\n[BLOCKED] {redact_secrets(str(exc))}")
        return 2
    except Exception as exc:
        print(f"\n[ERROR] Demo failed: {redact_secrets(str(exc))}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
