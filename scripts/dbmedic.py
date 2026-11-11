"""
scripts/dbmedic.py
==================
Unified Phase 6 CLI entry point for DBMedic.

Commands:
    python scripts/dbmedic.py doctor
    python scripts/dbmedic.py diagnose --incident slow_query_001 [--mode deterministic|tiny-aya|simulated]
    python scripts/dbmedic.py approve --incident slow_query_001
    python scripts/dbmedic.py verify --incident slow_query_001
    python scripts/dbmedic.py evaluate
    python scripts/dbmedic.py report

Exit codes:
    0   Operation succeeded cleanly
    1   Runtime / validation / state-machine error
    2   Environment or prerequisite blocked (e.g. DB offline, HF_TOKEN/GPU missing)
"""

from __future__ import annotations

import argparse
import getpass
import json
import logging
import sys
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env", override=False)

from agent.core.approval_manager import ApprovalError
from agent.core.audit_logger import AuditLogger, redact_secrets
from agent.core.environment_check import (
    STATUS_BLOCKED,
    STATUS_READY,
    check_tiny_aya_availability,
    generate_readiness_report,
)
from agent.core.incident_orchestrator import (
    EXECUTION_MODE_DETERMINISTIC,
    EXECUTION_MODE_LIVE_TINY_AYA,
    EXECUTION_MODE_SIMULATED,
    STATE_APPROVED,
    STATE_AWAITING_APPROVAL,
    STATE_DIAGNOSED,
    STATE_REJECTED,
    STATE_VERIFYING,
    IncidentNotFoundError,
    IncidentOrchestrator,
    InvalidStateTransitionError,
    OrchestrationBlockedError,
    normalize_execution_mode,
)
from agent.evaluation.final_evaluator import (
    generate_consolidated_report,
    run_unified_evaluation,
)
from agent.evaluation.incident_cases import get_case_by_id
from agent.tools.query_analyzer import DEFAULT_CUSTOMER_ID
from database.scenarios.slow_query import BENCHMARK_RUNS

KNOWN_INCIDENT_QUERIES: dict[str, dict[str, Any]] = {
    "slow_query_001": {
        "query": "Why is my orders query for customer_id = 4242 taking so long?",
        "customer_id": 4242,
    },
    "slow_query_002": {
        "query": "The query for customer 1234 on the orders table is very slow. Diagnose it.",
        "customer_id": 1234,
    },
    "slow_query_003": {
        "query": "Verify query performance on orders.customer_id = 4242.",
        "customer_id": 4242,
    },
}


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    if not verbose:
        for noisy in ("transformers", "torch", "huggingface_hub", "accelerate", "psycopg"):
            logging.getLogger(noisy).setLevel(logging.WARNING)


def _banner(title: str) -> None:
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------


def cmd_doctor(args: argparse.Namespace) -> int:
    _banner("DBMedic Environment Readiness Check (doctor)")
    report = generate_readiness_report(
        output_path=args.output,
        timeout_sec=args.timeout,
        save=True,
    )
    checks = report["checks"]
    caps = report["capabilities"]

    print(f"Overall Status          : {report['overall_status']}")
    print(f"Python Runtime          : [{checks['python_environment']['status']}] {checks['python_environment']['python_version']}")
    db = checks["database_connection"]
    db_target = db["target_database"].get("redacted_url") or "not configured"
    print(f"PostgreSQL (dbmedic_lab): [{db['status']}] {db_target}")
    schema = checks["lab_schema"]
    print(f"Lab Schema & Tables     : [{schema['status']}] present={schema['present_tables']}")
    if schema.get("orders_estimated_rows") is not None:
        print(
            f"Lab Orders Telemetry    : est_rows={schema['orders_estimated_rows']}, "
            f"idx_orders_customer_id_exists={schema['idx_orders_customer_id_exists']}"
        )
    hf = checks["huggingface_access"]
    print(f"HuggingFace Token/Cache : [{hf['status']}] token_present={hf['token_present']} cache_found={hf['local_model_cache_found']}")
    ta = checks["tiny_aya_global"]
    print(f"Tiny Aya Global         : [{ta['status']}] live_ready={ta['live_inference_ready']} checkpoint={ta['fine_tuned_checkpoint_exists']}")
    gpu = checks["gpu_and_memory"]
    print(f"GPU / RAM               : [{gpu['status']}] cuda={gpu['cuda_available']} free_ram={gpu['available_ram_gb']}GB/{gpu['total_ram_gb']}GB")
    at = checks["agenttune"]
    print(f"AgentTune               : [{at['status']}] version={at['version']}")
    ak = checks["auditkit"]
    print(f"AuditKIT                : [{ak['status']}] version={ak['version']} using_stub={ak['using_stub']}")

    print("\nCapabilities:")
    print(f"  Deterministic DB workflow : {caps['deterministic_db_mode_ready']}")
    print(f"  Live Tiny Aya AI workflow : {caps['full_live_ai_workflow_ready']}")
    print(f"  Offline evaluation / demo : {caps['offline_mode_ready']}")

    if report["blockers"]:
        print("\nBlockers Detected:")
        for b in report["blockers"]:
            print(f"  - {b}")

    print(f"\nReport saved to: {report.get('report_path')}")

    if args.json:
        print(json.dumps(redact_secrets(report), indent=2, default=str))

    if caps["live_postgres_ready"] and caps["python_core_ready"]:
        if args.require_live_ai and not caps["live_tiny_aya_ready"]:
            return 2
        return 0
    if args.allow_offline and caps["offline_mode_ready"]:
        return 0
    return 2


def _resolve_incident_prompt(incident_id: str, customer_override: int | None) -> tuple[str, int]:
    if incident_id in KNOWN_INCIDENT_QUERIES:
        entry = KNOWN_INCIDENT_QUERIES[incident_id]
        return entry["query"], int(customer_override or entry["customer_id"])
    case = get_case_by_id(incident_id)
    if case is not None:
        return case.query, int(customer_override or case.customer_id)
    return (
        f"Investigate slow query incident '{incident_id}' on orders.customer_id.",
        int(customer_override or DEFAULT_CUSTOMER_ID),
    )


def cmd_diagnose(args: argparse.Namespace) -> int:
    _banner(f"DBMedic Diagnose — Incident: {args.incident}")
    if args.mode is None:
        ta_status = check_tiny_aya_availability()
        if ta_status["live_inference_ready"]:
            selected_mode = EXECUTION_MODE_LIVE_TINY_AYA
            print("[MODE] Live Tiny Aya prerequisites satisfied -> execution_mode='live_tiny_aya'")
        else:
            selected_mode = EXECUTION_MODE_DETERMINISTIC
            print(
                "[MODE] Live Tiny Aya unavailable ("
                + "; ".join(ta_status["blockers"])
                + ") -> explicitly running execution_mode='deterministic'"
            )
    else:
        selected_mode = normalize_execution_mode(args.mode)
        print(f"[MODE] Explicit execution_mode='{selected_mode}'")

    if selected_mode == EXECUTION_MODE_SIMULATED:
        print("[NOTICE] SIMULATED MODE ENABLED — No live PostgreSQL or Tiny Aya inference executed.")

    query_desc, customer_id = _resolve_incident_prompt(args.incident, args.customer)
    orchestrator = IncidentOrchestrator()

    incident = orchestrator.create_incident(
        incident_id=args.incident,
        query_description=query_desc,
        customer_id=customer_id,
        execution_mode=selected_mode,
    )

    try:
        incident = orchestrator.collect_incident_evidence(
            incident,
            n_benchmark_runs=args.runs,
        )
        incident = orchestrator.run_agent_diagnosis(incident)
        incident = orchestrator.validate_diagnosis(incident)
    except OrchestrationBlockedError as exc:
        print(f"\n[BLOCKED] {exc}")
        print(f"Incident status: {incident['status']}")
        return 2
    except Exception as exc:
        print(f"\n[FAILED] Diagnostic investigation failed: {redact_secrets(str(exc))}")
        return 1

    diag = incident["diagnosis"]
    pre = incident["pre_remediation_metrics"]
    val = incident["diagnosis_validation"]

    print("\n--- Diagnostic Summary ---")
    print(f"Incident ID     : {incident['incident_id']}")
    print(f"Run ID          : {incident['run_id']}")
    print(f"Execution Mode  : {incident['execution_mode']} (simulated={incident['simulated']})")
    print(f"Model / Engine  : {incident['model_id']}")
    print(f"Scan Type       : {pre['scan_type']}")
    print(f"Baseline Median : {pre['median_execution_time_ms']} ms ({pre['n_runs']} runs)")
    print(f"Root Cause      : {diag['root_cause'].get('type')} — {diag['root_cause'].get('description')}")
    if diag.get("diagnostic_score") is not None:
        print(f"Diagnostic Score: {diag['diagnostic_score']}")
    print(f"Validation Pass : {val['valid']}")

    if val.get("requires_remediation"):
        incident = orchestrator.prepare_remediation(incident)
        incident = orchestrator.request_human_approval(incident, ttl_seconds=args.approval_ttl)
        req = incident["approval_request"]
        print("\n--- DBMedic Recommendation (Awaiting Human Approval) ---")
        print(f"Status          : {incident['status']}")
        print(f"Action ID       : {req['action_id']}")
        print(f"Target Database : {req['target_database'].get('database')} ({req['target_database'].get('redacted_url') or req['target_database'].get('host')})")
        print(f"Proposed SQL    : {req['proposed_sql']};")
        print(f"SQL SHA-256     : {req['sql_sha256']}")
        print(f"Expires At      : {req['expires_at']}")
        print("\nNo database modification has occurred.")
        print(f"Next step: python scripts/dbmedic.py approve --incident {incident['incident_id']}")
    else:
        incident = orchestrator.close_incident(incident)
        print(f"\nNo DDL remediation required. Incident closed cleanly as: {incident['status']}")

    if args.json:
        print(json.dumps(redact_secrets(incident), indent=2, default=str))
    return 0


def cmd_approve(args: argparse.Namespace) -> int:
    _banner(f"DBMedic Human Approval — Incident: {args.incident}")
    orchestrator = IncidentOrchestrator()
    try:
        incident = orchestrator.load_incident(args.incident)
    except IncidentNotFoundError as exc:
        print(f"[ERROR] {exc}")
        return 1

    if incident["status"] != STATE_AWAITING_APPROVAL:
        print(
            f"[ERROR] Incident '{args.incident}' is in state '{incident['status']}', "
            f"expected '{STATE_AWAITING_APPROVAL}'."
        )
        return 1

    req = incident.get("approval_request") or orchestrator.approval_manager.get_request(args.incident)
    if not req:
        print(f"[ERROR] No approval request payload found for '{args.incident}'.")
        return 1

    diag = incident.get("diagnosis") or {}
    pre = incident.get("pre_remediation_metrics") or {}

    print("DBMedic Recommendation Summary:")
    print(f"  Incident ID     : {incident['incident_id']}")
    print(f"  Execution Mode  : {incident['execution_mode']} (simulated={incident['simulated']})")
    print(f"  Action ID       : {req['action_id']}")
    print(f"  Target Database : {req['target_database'].get('database')} ({req['target_database'].get('redacted_url') or req['target_database'].get('host')})")
    print(f"  Root Cause      : {(diag.get('root_cause') or {}).get('description')}")
    print(f"  Baseline Plan   : {pre.get('scan_type')} | {pre.get('median_execution_time_ms')} ms")
    print(f"  Proposed SQL    : {req['proposed_sql']};")
    print(f"  SQL SHA-256     : {req['sql_sha256']}")
    print(f"  Expires At      : {req['expires_at']}")
    print("\nThe proposed statement is shown for review. No database modification has occurred.")

    decision = args.decision
    operator = args.operator
    reason = args.reason or ""

    if decision is None:
        if not sys.stdin.isatty():
            print(
                "\n[ERROR] Interactive terminal required for prompt, or supply explicit "
                "--decision approve --operator <name> --confirm-sql (or --decision reject)."
            )
            return 2

        if not operator:
            default_user = getpass.getuser() or "human_operator"
            entered_user = input(f"\nOperator name [{default_user}]: ").strip()
            operator = entered_user or default_user

        reply = input("Type 'APPROVE' to authorize this SQL statement, or 'REJECT' to decline: ").strip()
        if reply == "APPROVE":
            decision = "approve"
            confirm_method = "interactive_cli"
        elif reply.upper() == "REJECT":
            decision = "reject"
            reason = input("Rejection reason (optional): ").strip() or "Declined interactively"
            confirm_method = "interactive_cli"
        else:
            print("\nConfirmation did not match 'APPROVE'. Leaving incident in AWAITING_APPROVAL.")
            return 1
    else:
        if not operator:
            print("[ERROR] --operator <human_name> is required with --decision.")
            return 1
        if decision == "approve" and not args.confirm_sql:
            print("[ERROR] Explicit --confirm-sql flag is required with non-interactive --decision approve.")
            return 1
        confirm_method = "explicit_human_operator"

    try:
        incident = orchestrator.submit_human_decision(
            incident,
            approved=(decision == "approve"),
            operator=operator,
            confirmation_method=confirm_method,
            reason=reason,
        )
    except (ApprovalError, InvalidStateTransitionError) as exc:
        print(f"\n[REJECTED] Approval gate denied action: {exc}")
        return 1

    print(f"\nDecision recorded. Incident status is now: {incident['status']}")
    if incident["status"] == STATE_APPROVED:
        print(f"Next step: python scripts/dbmedic.py verify --incident {incident['incident_id']}")
    else:
        incident = orchestrator.close_incident(incident)
        print(f"Incident closed safely without DDL changes (status={incident['status']}).")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    _banner(f"DBMedic Execute & Verify Remediation — Incident: {args.incident}")
    orchestrator = IncidentOrchestrator()
    try:
        incident = orchestrator.load_incident(args.incident)
    except IncidentNotFoundError as exc:
        print(f"[ERROR] {exc}")
        return 1

    if incident["status"] == STATE_AWAITING_APPROVAL:
        print(
            f"[BLOCKED] Incident '{args.incident}' is awaiting human approval.\n"
            f"Run: python scripts/dbmedic.py approve --incident {args.incident}"
        )
        return 2
    if incident["status"] == STATE_REJECTED:
        print(f"[BLOCKED] Incident '{args.incident}' was rejected by operator and cannot be executed.")
        return 2

    if incident["status"] == STATE_APPROVED:
        print("Executing approved remediation DDL...")
        try:
            incident = orchestrator.execute_approved_remediation(incident)
        except (ApprovalError, InvalidStateTransitionError, Exception) as exc:
            print(f"[FAILED] Remediation blocked/failed: {redact_secrets(str(exc))}")
            return 1

    if incident["status"] != STATE_VERIFYING:
        print(
            f"[ERROR] Incident '{args.incident}' is in state '{incident['status']}', "
            f"expected '{STATE_APPROVED}' or '{STATE_VERIFYING}'."
        )
        return 1

    try:
        incident = orchestrator.verify_remediation(
            incident,
            n_benchmark_runs=args.runs,
        )
        incident = orchestrator.close_incident(incident)
    except Exception as exc:
        print(f"[FAILED] Verification error: {redact_secrets(str(exc))}")
        return 1

    cmp_rep = incident["verification_report"]
    before = cmp_rep["before"]
    after = cmp_rep["after"]
    perf = cmp_rep["performance_comparison"]
    outcome = cmp_rep["verification_outcome"]
    idx_v = cmp_rep["index_verification"]

    print("\n--- Remediation & Verification Results ---")
    print(f"Incident ID        : {incident['incident_id']}")
    print(f"Final Status       : {incident['status']}")
    print(f"Execution Mode     : {cmp_rep['execution_mode']} (simulated={cmp_rep['simulated']})")
    print(f"Verdict            : {outcome['verdict']} (success={outcome['success']})")
    print(f"Index Created      : {idx_v['index_exists']} ({idx_v.get('index_definition')})")
    print(f"Before Scan Type   : {before['scan_type']} | median={before['median_execution_time_ms']} ms")
    print(f"After Scan Type    : {after['scan_type']} | median={after['median_execution_time_ms']} ms")
    print(f"Speedup / Gain     : {perf['improvement_ms']} ms ({perf['improvement_pct']}% faster, {perf.get('speedup_factor')}x)")
    print(f"Buffer Hits/Reads  : Before ({before['shared_hit_blocks']}/{before['shared_read_blocks']}) -> After ({after['shared_hit_blocks']}/{after['shared_read_blocks']})")
    print(f"Explanation        : {outcome['planner_choice_explanation']}")
    print(f"\nComparison saved to: {cmp_rep.get('output_path')}")

    if args.json:
        print(json.dumps(redact_secrets(cmp_rep), indent=2, default=str))

    return 0 if outcome.get("success") or outcome.get("planner_choice_legitimate") else 1


def cmd_evaluate(args: argparse.Namespace) -> int:
    _banner("DBMedic Unified Evaluation (Phases 3–6)")
    metrics = run_unified_evaluation(run_redteam=not args.skip_redteam)

    ak = metrics["auditkit_status"]
    ep = metrics["episode_counts"]
    rem = metrics["remediation_metrics"]
    rt = metrics["safety_and_redteam"]
    gate = metrics["phase5_deployment_gate"]

    print(f"Evaluation Mode          : {metrics['evaluation_mode']}")
    print(f"AuditKIT Status          : {ak['validation_classification']} (version={ak['version']})")
    print(f"Real Model Episodes      : {ep['num_real_episodes']}")
    print(f"Simulated / Guard Cases  : {ep['num_simulated_episodes']} (never mixed with real metrics)")
    print(f"Red-Team Blocked         : {rt['attacks_blocked']}/{rt['attacks_tested']} (ASR={rt['attack_success_rate']})")
    print(f"Phase 6 Remediation      : before={rem['baseline_execution_time_ms']} ms -> after={rem['post_remediation_execution_time_ms']} ms ({rem['performance_improvement_pct']}%, mode={rem['execution_mode']})")
    print(f"Deployment Gate Decision : {gate['decision']} (approved={gate['approved']})")
    if gate.get("blockers"):
        print("Gate Blockers:")
        for b in gate["blockers"]:
            print(f"  - {b}")

    print(f"\nFinal metrics written to : {metrics.get('output_path')}")
    if args.json:
        print(json.dumps(redact_secrets(metrics), indent=2, default=str))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    _banner("DBMedic Consolidated Report Generator")
    if args.refresh:
        generate_readiness_report(save=True)
        run_unified_evaluation(run_redteam=True)

    paths = generate_consolidated_report(output_dir=args.output_dir)
    integrity = AuditLogger().verify_integrity()

    print(f"JSON Summary  : {paths['json_path']}")
    print(f"Markdown Summary: {paths['md_path']}")
    print(f"Audit Chain   : valid={integrity['valid']}, events={integrity['total_events']}")
    print(f"Audit Note    : {integrity['limitations']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dbmedic",
        description="DBMedic — Unified PostgreSQL Diagnostic & Safe Self-Healing CLI",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose debug logging.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_doc = sub.add_parser("doctor", help="Verify environment readiness (PostgreSQL, Tiny Aya, AgentTune, AuditKIT).")
    p_doc.add_argument("--output", default=None, help="Output JSON path (default: reports/environment_readiness.json).")
    p_doc.add_argument("--timeout", type=int, default=2, help="PostgreSQL connection timeout in seconds.")
    p_doc.add_argument("--allow-offline", action="store_true", help="Exit 0 when offline Python features are ready.")
    p_doc.add_argument("--require-live-ai", action="store_true", help="Require live Tiny Aya readiness for exit 0.")
    p_doc.add_argument("--json", action="store_true", help="Print full JSON report to stdout.")

    p_diag = sub.add_parser("diagnose", help="Diagnose a PostgreSQL incident cleanly.")
    p_diag.add_argument("--incident", default="slow_query_001", help="Incident ID (default: slow_query_001).")
    p_diag.add_argument(
        "--mode",
        choices=["deterministic", "tiny-aya", "simulated"],
        default=None,
        help="Explicit execution mode. Omit to auto-check Tiny Aya vs explicit deterministic.",
    )
    p_diag.add_argument("--customer", type=int, default=None, help="Override target customer_id.")
    p_diag.add_argument("--runs", type=int, default=BENCHMARK_RUNS, help="Baseline benchmark iterations (default: 7).")
    p_diag.add_argument("--approval-ttl", type=int, default=1800, help="Approval expiration in seconds (default: 1800).")
    p_diag.add_argument("--json", action="store_true", help="Print full sanitized incident JSON.")

    p_app = sub.add_parser("approve", help="Review and explicitly approve or reject proposed DDL.")
    p_app.add_argument("--incident", default="slow_query_001", help="Incident ID awaiting approval.")
    p_app.add_argument("--decision", choices=["approve", "reject"], default=None, help="Explicit non-interactive decision.")
    p_app.add_argument("--operator", default=None, help="Human operator name (model identifiers forbidden).")
    p_app.add_argument("--confirm-sql", action="store_true", help="Confirm operator reviewed exact SQL statement.")
    p_app.add_argument("--reason", default="", help="Optional rejection reason.")

    p_ver = sub.add_parser("verify", help="Execute approved remediation and verify before/after improvement.")
    p_ver.add_argument("--incident", default="slow_query_001", help="Incident ID to execute & verify.")
    p_ver.add_argument("--runs", type=int, default=BENCHMARK_RUNS, help="Benchmark iterations (default: 7).")
    p_ver.add_argument("--json", action="store_true", help="Print full verification comparison JSON.")

    p_eval = sub.add_parser("evaluate", help="Connect Phases 3–6 evaluation and AuditKIT adapter.")
    p_eval.add_argument("--skip-redteam", action="store_true", help="Reuse cached redteam_results.json instead of running.")
    p_eval.add_argument("--json", action="store_true", help="Print full final metrics JSON.")

    p_rep = sub.add_parser("report", help="Generate consolidated JSON + Markdown Phase 6 summary.")
    p_rep.add_argument("--output-dir", default=None, help="Directory for final_summary.* (default: reports/).")
    p_rep.add_argument("--refresh", action="store_true", help="Refresh readiness & evaluation before reporting.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)

    dispatch = {
        "doctor": cmd_doctor,
        "diagnose": cmd_diagnose,
        "approve": cmd_approve,
        "verify": cmd_verify,
        "evaluate": cmd_evaluate,
        "report": cmd_report,
    }
    handler = dispatch[args.command]
    return int(handler(args))


if __name__ == "__main__":
    sys.exit(main())

