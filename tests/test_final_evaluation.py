"""
tests/test_final_evaluation.py
==============================
Tests connecting Phases 3–6 evaluation, CLI (`scripts/dbmedic.py`), demo scripts,
and Streamlit dashboard panel rendering without mixing synthetic and live metrics.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent.core.audit_logger import AuditLogger
from agent.evaluation.final_evaluator import (
    generate_consolidated_report,
    run_unified_evaluation,
)
from dashboard.components.incident_panel import summarize_database_health
from demo.scripts.run_complete_demo import run_demo
from scripts.dbmedic import main as dbmedic_cli_main


@pytest.mark.unit
def test_unified_evaluation_honest_nulls_and_stub_disclosure(tmp_path: Path):
    out_file = tmp_path / "final_metrics.json"
    audit = AuditLogger(tmp_path / "audit.jsonl")

    metrics = run_unified_evaluation(
        project_root=PROJECT_ROOT,
        output_path=out_file,
        run_redteam=True,
        audit_logger=audit,
    )
    assert out_file.exists()

    # Live model metrics must remain null when no real inference occurred
    live_m = metrics["agent_metrics_live_model"]
    if metrics["episode_counts"]["num_real_episodes"] == 0:
        for key in (
            "diagnosis_accuracy",
            "tool_call_validity_rate",
            "tool_selection_accuracy",
            "task_success_rate",
            "average_tool_calls_per_task",
            "average_steps_per_task",
            "safety_violation_count",
            "evidence_grounding_accuracy",
        ):
            assert live_m[key] is None

    # Stub must be explicitly distinguished from independent AuditKIT validation
    ak = metrics["auditkit_status"]
    if ak["using_stub"]:
        assert "NOT independent AuditKIT validation" in ak["validation_classification"]

    # Deterministic red-team suite runs and blocks 10/10
    rt = metrics["safety_and_redteam"]
    assert rt["attacks_tested"] == 10
    assert rt["attacks_blocked"] == 10
    assert rt["attack_success_rate"] == 0.0

    # Real vs simulated episode counts are separated
    assert metrics["episode_counts"]["num_simulated_episodes"] > 0
    assert metrics["phase5_deployment_gate"]["decision"] == "hold_keep_base"


@pytest.mark.unit
def test_cli_doctor_and_evaluate_exit_and_outputs(tmp_path: Path, monkeypatch):
    # CLI doctor --allow-offline works even when DB offline
    out_readiness = tmp_path / "readiness.json"
    rc_doc = dbmedic_cli_main(["doctor", "--output", str(out_readiness), "--allow-offline"])
    assert rc_doc == 0
    assert out_readiness.exists()
    data = json.loads(out_readiness.read_text(encoding="utf-8"))
    assert "overall_status" in data
    assert "auditkit" in data["checks"]


@pytest.mark.unit
def test_simulated_demo_workflow_isolated(tmp_path: Path):
    summary = run_demo(
        scenario_name="slow_query_missing_index",
        mode="simulated",
        operator="judge_operator",
        confirm_human_approval=True,
        n_runs=3,
        incidents_dir=tmp_path / "incidents",
        approval_store_path=tmp_path / "approvals.json",
        audit_log_path=tmp_path / "audit_trail.jsonl",
        comparison_output_path=tmp_path / "remediation_comparison.json",
        final_metrics_path=tmp_path / "final_metrics.json",
        report_output_dir=tmp_path / "reports",
    )
    inc = summary["incident"]
    assert inc["status"] == "RESOLVED"
    assert inc["simulated"] is True
    assert inc["execution_mode"] == "simulated"
    assert (tmp_path / "remediation_comparison.json").exists()
    assert (tmp_path / "final_metrics.json").exists()
    assert (tmp_path / "audit_trail.jsonl").exists()


@pytest.mark.unit
def test_consolidated_report_and_dashboard_panel_helpers(tmp_path: Path):
    rep = generate_consolidated_report(project_root=PROJECT_ROOT, output_dir=tmp_path)
    assert Path(rep["json_path"]).exists()
    assert Path(rep["md_path"]).exists()

    # Dashboard import & health sanitizer safety
    health = summarize_database_health(None)
    assert health["connected"] is False

    import dashboard.app  # noqa: F401
    import dashboard.components.approval_panel  # noqa: F401
    import dashboard.components.audit_panel  # noqa: F401
    import dashboard.components.benchmark_panel  # noqa: F401
    import dashboard.components.diagnostic_panel  # noqa: F401

