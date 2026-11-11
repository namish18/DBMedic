"""
tests/test_remediation_verifier.py
==================================
Tests for before/after remediation metrics capture, plan comparison,
legitimate sequential scan explanations, and remediation_comparison.json output.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent.core.remediation_verifier import (
    calculate_performance_improvement,
    capture_post_remediation_metrics,
    capture_pre_remediation_metrics,
    compare_execution_plans,
    determine_remediation_success,
    verify_and_save_comparison,
    verify_index_created,
)


@pytest.mark.unit
def test_remediation_verifier_captures_all_metrics_and_saves_comparison(tmp_path: Path):
    before = capture_pre_remediation_metrics(customer_id=4242, n_runs=7, simulated=True)
    after = capture_post_remediation_metrics(customer_id=4242, n_runs=7, simulated=True)
    idx_v = verify_index_created(simulated=True)

    for key in (
        "execution_time_ms",
        "median_execution_time_ms",
        "planning_time_ms",
        "scan_type",
        "actual_rows",
        "estimated_rows",
        "shared_hit_blocks",
        "shared_read_blocks",
        "index_exists",
        "index_definition",
    ):
        assert key in before
        assert key in after

    out_path = tmp_path / "remediation_comparison.json"
    report = verify_and_save_comparison(
        incident_id="slow_query_001",
        before_metrics=before,
        after_metrics=after,
        index_verification=idx_v,
        execution_mode="simulated",
        output_path=out_path,
    )

    assert out_path.exists()
    assert report["simulated"] is True
    assert report["verification_outcome"]["success"] is True
    assert report["verification_outcome"]["verdict"] == "VERIFIED_IMPROVED"
    assert report["plan_comparison"]["switched_from_seq_to_index"] is True
    assert report["performance_comparison"]["improvement_pct"] > 80.0


@pytest.mark.unit
def test_verifier_does_not_declare_success_solely_from_index_existence():
    before = capture_pre_remediation_metrics(customer_id=4242, n_runs=5, simulated=True)
    # Force post metrics where index exists, yet plan stayed Seq Scan on low selectivity filter
    after = {
        **before,
        "phase": "post_remediation",
        "median_execution_time_ms": 23.8,
        "index_exists": True,
        "index_definition": "CREATE INDEX idx_orders_customer_id ON public.orders USING btree (customer_id)",
        "actual_rows": 60_000,
        "table_row_count": 200_000,
    }
    idx_v = verify_index_created(simulated=True)
    plan_cmp = compare_execution_plans(before, after)
    perf_cmp = calculate_performance_improvement(before, after)
    outcome = determine_remediation_success(before, after, idx_v, plan_cmp, perf_cmp)

    assert outcome["success"] is False
    assert outcome["verdict"] == "PLANNER_LEGITIMATE_SEQ_SCAN_LOW_SELECTIVITY"
    assert outcome["planner_choice_legitimate"] is True
    assert "sequential i/o" in outcome["planner_choice_explanation"].lower()


@pytest.mark.unit
def test_verifier_explains_small_table_seq_scan():
    before = capture_pre_remediation_metrics(customer_id=4242, n_runs=3, simulated=True)
    after = {
        **before,
        "phase": "post_remediation",
        "median_execution_time_ms": 0.4,
        "index_exists": True,
        "actual_rows": 5,
        "table_row_count": 250,
    }
    idx_v = verify_index_created(simulated=True)
    plan_cmp = compare_execution_plans(before, after)
    perf_cmp = calculate_performance_improvement(before, after)
    outcome = determine_remediation_success(before, after, idx_v, plan_cmp, perf_cmp)

    assert outcome["success"] is False
    assert outcome["verdict"] == "PLANNER_LEGITIMATE_SEQ_SCAN_SMALL_TABLE"
    assert outcome["planner_choice_legitimate"] is True
    assert "250 rows" in outcome["planner_choice_explanation"]


@pytest.mark.unit
def test_verifier_rejects_incomparable_workloads():
    before = capture_pre_remediation_metrics(customer_id=4242, n_runs=3, simulated=True)
    after = capture_post_remediation_metrics(customer_id=9999, n_runs=3, simulated=True)
    idx_v = verify_index_created(simulated=True)
    plan_cmp = compare_execution_plans(before, after)
    perf_cmp = calculate_performance_improvement(before, after)
    outcome = determine_remediation_success(before, after, idx_v, plan_cmp, perf_cmp)

    assert outcome["success"] is False
    assert outcome["verdict"] == "INCOMPARABLE_WORKLOAD"

