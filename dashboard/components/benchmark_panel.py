"""
dashboard/components/benchmark_panel.py
=======================================
Before/after PostgreSQL benchmark comparison and evaluation metrics panel.
"""

from __future__ import annotations

from typing import Any


def render_benchmark_panel(
    st: Any,
    comparison: dict[str, Any] | None,
    final_metrics: dict[str, Any] | None,
) -> None:
    """Render Before/After Performance & Agent Evaluation Metrics sections."""
    st.subheader("6. Before / After Remediation Performance")
    if not comparison:
        st.info("No Phase 6 remediation comparison recorded yet (`benchmarks/final_demo/remediation_comparison.json`).")
    else:
        if comparison.get("simulated"):
            st.warning("SIMULATED BENCHMARK DATA — Not measured on live PostgreSQL.")
        before = comparison.get("before", {})
        after = comparison.get("after", {})
        perf = comparison.get("performance_comparison", {})
        outcome = comparison.get("verification_outcome", {})

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Before Latency", f"{before.get('median_execution_time_ms')} ms", str(before.get("scan_type")))
        c2.metric("After Latency", f"{after.get('median_execution_time_ms')} ms", str(after.get("scan_type")))
        c3.metric("Improvement", f"{perf.get('improvement_pct')}%", f"{perf.get('speedup_factor')}x speedup")
        c4.metric("Verdict", str(outcome.get("verdict")))
        st.write(f"**Verification Explanation:** {outcome.get('planner_choice_explanation')}")

    st.subheader("7. Agent Evaluation & AuditKIT Status")
    if not final_metrics:
        st.info("Run `python scripts/dbmedic.py evaluate` to inspect unified Phase 3–6 metrics.")
        return

    ak = final_metrics.get("auditkit_status", {})
    ep = final_metrics.get("episode_counts", {})
    rt = final_metrics.get("safety_and_redteam", {})
    gate = final_metrics.get("phase5_deployment_gate", {})

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("AuditKIT Mode", "Real" if not ak.get("using_stub") else "Offline Stub")
    m2.metric("Real Model Episodes", str(ep.get("num_real_episodes", 0)))
    m3.metric("Red-Team Blocked", f"{rt.get('attacks_blocked')}/{rt.get('attacks_tested')}")
    m4.metric("Deployment Gate", str(gate.get("decision", "unknown")))

    st.write(f"**AuditKIT Classification:** `{ak.get('validation_classification')}`")
    with st.expander("Live Tiny Aya Model Metrics (Pending Values Remain Null)"):
        st.json(final_metrics.get("agent_metrics_live_model"))

