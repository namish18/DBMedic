"""
tests/test_training_dataset.py
==============================
Unit tests for Phase 4 dataset curation, leakage isolation, Tiny Aya strict
alternation formatting, SHA-256 integrity, and token-level -100 completion masking.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent.evaluation.incident_cases import BASELINE_EVAL_CASES, HIDDEN_EVAL_CASES, get_case_by_id
from training.data.curator import DatasetCurator, compute_file_sha256
from training.data.formatter import (
    enforce_strict_alternation,
    render_tiny_aya_chat,
    tokenize_and_mask_sft_row,
    validate_alternation,
)
from training.data.validator import validate_dataset_artifacts, verify_no_data_leakage
from training.failure_analysis.correction_generator import CorrectionGenerator


class _DeterministicDummyTokenizer:
    """Minimal byte-level tokenizer supporting Tiny Aya Jinja rendering for unit tests."""

    chat_template = (
        "<BOS_TOKEN>"
        "{%- for m in messages -%}"
        "{%- if m['role'] == 'system' -%}"
        "<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|># Developer Preamble\n{{ m['content'] }}<|END_OF_TURN_TOKEN|>"
        "{%- elif m['role'] == 'user' -%}"
        "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{{ m['content'] }}<|END_OF_TURN_TOKEN|>"
        "{%- else -%}"
        "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_RESPONSE|>{{ m['content'] }}<|END_RESPONSE|><|END_OF_TURN_TOKEN|>"
        "{%- endif -%}"
        "{%- endfor -%}"
        "{%- if add_generation_prompt -%}"
        "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_RESPONSE|>"
        "{%- endif -%}"
    )

    pad_token_id = 0
    eos_token_id = 1
    pad_token = "<pad>"
    eos_token = "<eos>"

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ) -> str:
        # Strict Tiny Aya role alternation enforcement
        start = 1 if messages and messages[0]["role"] == "system" else 0
        expected = "user"
        parts = ["<BOS_TOKEN>"]
        if start == 1:
            parts.append(f"<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|># Developer Preamble\n{messages[0]['content']}<|END_OF_TURN_TOKEN|>")
        for m in messages[start:]:
            if m["role"] != expected:
                raise ValueError("Conversation roles must alternate user/assistant.")
            if m["role"] == "user":
                parts.append(f"<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{m['content']}<|END_OF_TURN_TOKEN|>")
                expected = "assistant"
            else:
                parts.append(
                    f"<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_RESPONSE|>{m['content']}<|END_RESPONSE|><|END_OF_TURN_TOKEN|>"
                )
                expected = "user"
        if add_generation_prompt:
            parts.append("<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_RESPONSE|>")
        text = "".join(parts)
        return self(text)["input_ids"] if tokenize else text

    def __call__(self, text: str, add_special_tokens: bool = False, **kwargs) -> dict[str, list[int]]:
        # Encode each character consistently as an integer ID >= 10
        return {"input_ids": [10 + (ord(ch) % 240) for ch in text]}


@pytest.mark.unit
class TestTrainingDataset:
    def test_curator_builds_verified_deduped_leakage_free_splits(self, tmp_path: Path):
        # Include an unsafe/unverified candidate alongside duplicate valid examples
        bad_unsafe_row = {
            "incident_id": "train_001",
            "dataset_split": "train",
            "query": "Why slow?",
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "Why slow?"},
                {"role": "assistant", "content": "DROP TABLE orders;"},
            ],
            "tools_called": ["apply_remediation"],
            "diagnosis": "Dropped orders",
            "root_cause": "missing_suitable_index",
            "evidence": ["none"],
            "recommended_action": "DROP TABLE orders",
            "requires_approval": False,
            "success": False,
            "safety_violations": ["Attempted apply_remediation"],
            "provenance": {"correction_source": "verified_success_trajectory"},
        }
        # Include a hidden incident attempting to sneak into train split
        leaked_hidden_row = {
            "incident_id": HIDDEN_EVAL_CASES[0].incident_id,
            "scenario_group": HIDDEN_EVAL_CASES[0].scenario_group,
            "dataset_split": "train",
            "query": HIDDEN_EVAL_CASES[0].query,
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": HIDDEN_EVAL_CASES[0].query},
                {"role": "assistant", "content": '{"type": "final_answer", "diagnosis": "test"}'},
            ],
            "diagnosis": "Leaked hidden case",
            "root_cause": "missing_suitable_index",
            "provenance": {"correction_source": "verified_success_trajectory"},
        }

        curator = DatasetCurator()
        res = curator.build(
            extra_trajectories=[bad_unsafe_row],
            extra_records=[leaked_hidden_row],
            include_controlled_demonstrations=True,
            output_root=tmp_path,
        )

        # Unsafe bad trajectory gets repaired cleanly into 1 train_001 example, deduped against demo!
        assert res.duplicates_removed >= 1
        # Hidden leaked record explicitly rejected!
        assert res.rejected_count >= 1
        assert res.train_count == 18
        assert res.validation_count == 5
        assert res.hidden_test_count == 5

        report = validate_dataset_artifacts(tmp_path)
        assert report.passed, f"Validation errors: {report.errors}"
        assert report.sha256_verified
        assert report.leakage_clean
        assert report.safety_clean
        assert report.protocol_clean

    def test_hidden_test_jsonl_strips_all_expected_answers(self, tmp_path: Path):
        DatasetCurator().build(include_controlled_demonstrations=True, output_root=tmp_path)
        hidden_lines = (tmp_path / "splits" / "hidden_test.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(hidden_lines) == 5
        for line in hidden_lines:
            row = json.loads(line)
            assert "expected_root_cause" not in row
            assert "expected_tools_to_call" not in row
            assert "expected_action" not in row
            assert "messages" not in row
            assert "query" in row

    def test_validator_catches_tampered_sha256_and_leakage(self, tmp_path: Path):
        DatasetCurator().build(include_controlled_demonstrations=True, output_root=tmp_path)
        train_path = tmp_path / "splits" / "train.jsonl"
        manifest_path = tmp_path / "dataset_manifest.json"
        # Tamper manifest hash
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["sha256"]["train_jsonl"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        report = validate_dataset_artifacts(tmp_path)
        assert not report.passed
        assert not report.sha256_verified

        # Explicit leakage check against baseline & hidden cases
        ok, errs = verify_no_data_leakage(
            [{"incident_id": BASELINE_EVAL_CASES[0].incident_id, "scenario_group": BASELINE_EVAL_CASES[0].scenario_group, "query": "x"}],
            [],
            [{"incident_id": "h1", "scenario_group": "g1", "query": "q1"}],
        )
        assert not ok and errs

    def test_strict_alternation_and_token_level_completion_masking(self):
        msgs = [
            {"role": "system", "content": "SYSTEM PROMPT"},
            {"role": "user", "content": "USER QUESTION"},
            {"role": "assistant", "content": '{"type":"tool_call","tool":"get_table_statistics","arguments":{"table_name":"orders"}}'},
            {"role": "tool", "name": "get_table_statistics", "content": '{"row_count": 200000}'},
            {"role": "assistant", "content": '{"type":"final_answer","diagnosis":"Seq scan on 200000 rows.","root_cause":"missing_suitable_index","requires_approval":true}'},
        ]
        strict = enforce_strict_alternation(msgs)
        ok, err = validate_alternation(strict)
        assert ok, err
        assert [m["role"] for m in strict] == ["system", "user", "assistant", "user", "assistant"]

        tok = _DeterministicDummyTokenizer()
        rendered = render_tiny_aya_chat(strict, tokenizer=tok)
        assert "<|SYSTEM_TOKEN|>" in rendered
        assert "<|START_RESPONSE|>" in rendered

        masked = tokenize_and_mask_sft_row(strict, tok, max_seq_length=4096)
        assert len(masked["input_ids"]) == len(masked["labels"]) == len(masked["assistant_mask"])
        assert 0 in masked["assistant_mask"]
        assert 1 in masked["assistant_mask"]
        # Verify system and user/tool turns carry -100 while assistant tokens match input_ids
        for tid, amask, lbl in zip(masked["input_ids"], masked["assistant_mask"], masked["labels"]):
            if amask == 0:
                assert lbl == -100
            else:
                assert lbl == tid

