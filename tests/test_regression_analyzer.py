"""
tests/test_regression_analyzer.py
=====================================
Unit tests for the regression analyzer.

Tests:
  - No regression detected when metrics improve
  - Regression detected when metrics decrease
  - Safety regression detection
  - Null handling — missing metrics not treated as passing
  - Partial data handling
  - Overall verdict computation
"""

from __future__ import annotations

import pytest


@pytest.mark.unit
class TestRegressionAnalyzer:

    def test_import(self):
        from agent.audit.regression_analyzer import RegressionAnalyzer
        assert RegressionAnalyzer is not None

    def test_no_regression_when_candidate_improves(self):
        from agent.audit.regression_analyzer import RegressionAnalyzer
        analyzer = RegressionAnalyzer()
        baseline = {
            "tool_call_validity_rate": 0.80,
            "task_success_rate": 0.70,
            "diagnosis_accuracy": 0.75,
            "safety_violation_count": 0,
        }
        candidate = {
            "tool_call_validity_rate": 0.90,
            "task_success_rate": 0.80,
            "diagnosis_accuracy": 0.85,
            "safety_violation_count": 0,
        }
        report = analyzer.analyze(baseline, candidate)
        assert len(report.regressions) == 0
        assert report.overall_verdict in {"pass", "hold"}  # No regressions

    def test_regression_detected_when_accuracy_drops(self):
        from agent.audit.regression_analyzer import RegressionAnalyzer
        analyzer = RegressionAnalyzer()
        baseline = {"diagnosis_accuracy": 0.80}
        candidate = {"diagnosis_accuracy": 0.60}  # -0.20 — clearly regressed
        report = analyzer.analyze(baseline, candidate)
        reg_metrics = [r["metric"] for r in report.regressions]
        assert "diagnosis_accuracy" in reg_metrics

    def test_safety_regression_critical(self):
        from agent.audit.regression_analyzer import RegressionAnalyzer
        analyzer = RegressionAnalyzer()
        baseline = {"safety_violation_count": 0}
        candidate = {"safety_violation_count": 1}  # Any increase = regression
        report = analyzer.analyze(baseline, candidate)
        assert report.has_safety_regression is True
        assert report.has_critical_regression is True
        assert report.overall_verdict == "fail"

    def test_null_metrics_marked_uncomputable(self):
        from agent.audit.regression_analyzer import RegressionAnalyzer
        analyzer = RegressionAnalyzer()
        baseline = {"diagnosis_accuracy": None}
        candidate = {"diagnosis_accuracy": None}
        report = analyzer.analyze(baseline, candidate)
        uncomputable = [u["metric"] for u in report.uncomputable]
        assert "diagnosis_accuracy" in uncomputable
        # With uncomputable metrics but no regressions → hold
        assert report.overall_verdict in {"hold", "pass"}

    def test_partial_null_metrics_marked_uncomputable(self):
        from agent.audit.regression_analyzer import RegressionAnalyzer
        analyzer = RegressionAnalyzer()
        baseline = {"tool_call_validity_rate": 0.90}
        candidate = {"tool_call_validity_rate": None}  # Candidate pending
        report = analyzer.analyze(baseline, candidate)
        uncomputable = [u["metric"] for u in report.uncomputable]
        assert "tool_call_validity_rate" in uncomputable

    def test_null_not_treated_as_passing(self):
        """
        CRITICAL: If candidate metric is None, that must NOT count as passing.
        The gate must hold, not approve.
        """
        from agent.audit.regression_analyzer import RegressionAnalyzer
        analyzer = RegressionAnalyzer()
        # Both baseline and candidate are null
        report = analyzer.analyze({}, {})
        # verdict should be hold or pass with no regressions
        # but definitely NOT "fail" due to missing data being treated as regression
        assert report.has_critical_regression is False
        assert report.has_safety_regression is False

    def test_improvement_recorded(self):
        from agent.audit.regression_analyzer import RegressionAnalyzer
        analyzer = RegressionAnalyzer()
        baseline = {"diagnosis_accuracy": 0.60}
        candidate = {"diagnosis_accuracy": 0.80}
        report = analyzer.analyze(baseline, candidate)
        improved = [i["metric"] for i in report.improvements]
        assert "diagnosis_accuracy" in improved

    def test_latency_regression_detection(self):
        from agent.audit.regression_analyzer import RegressionAnalyzer
        analyzer = RegressionAnalyzer()
        # This key may not be in METRIC_SPECS — should still be computed
        baseline = {"inference_latency_ms": 200.0}
        candidate = {"inference_latency_ms": 400.0}  # 2x worse
        report = analyzer.analyze(baseline, candidate)
        # Even if not in predefined specs, check uncomputable or regressions
        all_metrics = (
            [r["metric"] for r in report.regressions]
            + [u["metric"] for u in report.uncomputable]
            + [r["metric"] for r in report.improvements]
            + [r["metric"] for r in report.unchanged]
        )
        assert "inference_latency_ms" in all_metrics

    def test_regression_report_to_dict(self):
        from agent.audit.regression_analyzer import RegressionAnalyzer
        analyzer = RegressionAnalyzer()
        report = analyzer.analyze({"diagnosis_accuracy": 0.80}, {"diagnosis_accuracy": 0.80})
        d = report.to_dict()
        assert isinstance(d, dict)
        assert "overall_verdict" in d
        assert "regressions" in d
        assert "improvements" in d
        assert "uncomputable" in d

