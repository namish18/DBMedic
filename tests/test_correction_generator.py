"""
tests/test_correction_generator.py
==================================
Unit tests for DBMedic Phase 4 CorrectionGenerator & independent verification gates.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent.evaluation.incident_cases import TRAINING_CANDIDATES, get_case_by_id
from training.data.formatter import validate_alternation
from training.failure_analysis.classifier import FailureClassifier, FailureRecord
from training.failure_analysis.correction_generator import CorrectionGenerator


@pytest.mark.unit
class TestCorrectionGenerator:
    def test_generate_for_every_training_case_passes_independent_verifier(self):
        gen = CorrectionGenerator()
        clf = FailureClassifier()
        for case in TRAINING_CANDIDATES:
            rec = gen.generate_for_case(case, correction_source="human_reviewed_demonstration")
            ver = gen.verify_correction(rec, expected_case=case)
            assert ver.verified, f"{case.incident_id} failed verification: {ver.reasons}"
            alt_ok, alt_err = validate_alternation(rec["messages"])
            assert alt_ok, f"{case.incident_id} alternation failed: {alt_err}"
            # Repaired trajectory must also pass FailureClassifier with zero failures!
            assert clf.classify(rec, case=case) == []

    def test_generate_from_failure_record_produces_verified_repair(self):
        gen = CorrectionGenerator()
        case = get_case_by_id("train_006")
        fr = FailureRecord(
            incident_id="train_006",
            failure_category="incorrect_root_cause",
            original_model_response='{"type": "final_answer", "root_cause": "cpu_throttling"}',
            expected_behavior="Diagnose missing_suitable_index",
            observed_behavior="Wrong root cause",
            evidence=["Expected missing_suitable_index"],
            trajectory_id="traj_bad_006",
        )
        repaired = gen.generate_from_failure(fr, case=case)
        ver = gen.verify_correction(repaired, expected_case=case)
        assert ver.verified, f"Repaired trace failed verifier: {ver.reasons}"
        assert repaired["provenance"]["correction_source"] == "verified_corrected_failure"
        assert repaired["provenance"]["original_trajectory_id"] == "traj_bad_006"

    def test_safety_adversarial_correction_teaches_safe_refusal(self):
        gen = CorrectionGenerator()
        adv_case = get_case_by_id("train_016")
        assert adv_case is not None and adv_case.safety_adversarial
        rec = gen.generate_for_case(adv_case)
        ver = gen.verify_correction(rec, expected_case=adv_case)
        assert ver.verified
        assert rec["requires_approval"] is True
        action_lower = rec["recommended_action"].lower()
        assert "approval" in action_lower or "read-only" in action_lower
        assert "apply_remediation" not in rec["tools_called"]

    def test_verifier_rejects_unapproved_ddl(self):
        gen = CorrectionGenerator()
        case = get_case_by_id("train_001")
        rec = gen.generate_for_case(case)
        rec["requires_approval"] = False
        last = json.loads(rec["messages"][-1]["content"])
        last["requires_approval"] = False
        rec["messages"][-1]["content"] = json.dumps(last)
        ver = gen.verify_correction(rec, expected_case=case)
        assert not ver.verified

    def test_verifier_rejects_wrong_root_cause_or_sql_injection(self):
        gen = CorrectionGenerator()
        case = get_case_by_id("train_001")
        rec = gen.generate_for_case(case)
        last = json.loads(rec["messages"][-1]["content"])
        last["root_cause"] = "wrong_cause"
        rec["messages"][-1]["content"] = json.dumps(last)
        ver = gen.verify_correction(rec, expected_case=case)
        assert not ver.verified

        rec2 = gen.generate_for_case(case)
        rec2["messages"][2]["content"] = json.dumps(
            {"type": "tool_call", "tool": "get_table_statistics", "arguments": {"table_name": "orders; DROP TABLE orders;"}}
        )
        ver2 = gen.verify_correction(rec2, expected_case=case)
        assert not ver2.verified

