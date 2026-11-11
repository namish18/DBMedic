"""
tests/test_training_config.py
=============================
Unit tests for Phase 4 YAML training configs, LoRA/TRL config builders,
readiness preflight checks, GRPO gating & read-only tools, and tiny-model
TRL SFTTrainer completion-masked training smoke verification.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.agenttune.distillation import (
    DBMedicSFTTrainerFactory,
    build_lora_config,
    build_sft_config,
    check_training_readiness,
)
from training.agenttune.grpo_training import (
    build_readonly_grpo_tools,
    check_grpo_gating,
    run_grpo_experiment,
)
from training.data.curator import DatasetCurator
from training.failure_analysis.classifier import APPROVED_DIAGNOSTIC_TOOLS


@pytest.mark.unit
class TestTrainingConfig:
    def test_yaml_configs_load_valid_and_enforce_safety(self):
        ds_cfg = yaml.safe_load((PROJECT_ROOT / "training/configs/dataset_config.yaml").read_text(encoding="utf-8"))
        sft_cfg = yaml.safe_load((PROJECT_ROOT / "training/configs/sft_config.yaml").read_text(encoding="utf-8"))
        grpo_cfg = yaml.safe_load((PROJECT_ROOT / "training/configs/grpo_config.yaml").read_text(encoding="utf-8"))

        assert ds_cfg["splits"]["strip_hidden_answers_from_prompts"] is True
        assert sft_cfg["model"]["base_model_id"] == "CohereLabs/tiny-aya-global"
        assert sft_cfg["peft"]["target_modules"] == "all-linear"
        assert "lm_head" in (sft_cfg["peft"].get("exclude_modules") or [])
        assert sft_cfg["training"]["loss_type"] == "nll"

        assert grpo_cfg["experiment"]["enabled"] is False
        assert grpo_cfg["rollout"]["read_only_tools_only"] is True
        assert "apply_remediation" in grpo_cfg["rollout"]["excluded_tools"]

    def test_lora_and_sft_configs_build_cleanly_for_trl_and_peft(self, tmp_path: Path):
        sft_cfg = yaml.safe_load((PROJECT_ROOT / "training/configs/sft_config.yaml").read_text(encoding="utf-8"))
        lora = build_lora_config(sft_cfg)
        assert lora.r == 16
        assert lora.lora_alpha == 32
        assert lora.target_modules == "all-linear" or "all-linear" in str(lora.target_modules)

        trl_cfg = build_sft_config(sft_cfg, output_dir=tmp_path / "sft_out", cuda_available=False)
        assert trl_cfg.loss_type == "nll"
        assert trl_cfg.completion_only_loss is False
        assert trl_cfg.packing is False

    def test_readonly_grpo_tools_strictly_exclude_mutations(self):
        tools = build_readonly_grpo_tools()
        names = {t.name for t in tools}
        assert names == set(APPROVED_DIAGNOSTIC_TOOLS)
        assert "apply_remediation" not in names
        assert "execute_sql" not in names
        with pytest.raises(PermissionError):
            build_readonly_grpo_tools(allowed_tools=["apply_remediation"])

    def test_grpo_gating_blocks_cleanly_without_prerequisites(self, tmp_path: Path):
        gating = check_grpo_gating(sft_checkpoint_dir=tmp_path / "missing_sft", check_db=False)
        assert gating.passed is False
        assert gating.sft_checkpoint_exists is False
        assert len(gating.blockers) >= 1

        res = run_grpo_experiment(output_dir=tmp_path / "grpo_out", dry_run=False, check_db=False)
        assert res["status"] == "gated_blocked"
        assert res["executed_training"] is False

    def test_preflight_readiness_runs_safely_offline(self, tmp_path: Path):
        DatasetCurator().build(include_controlled_demonstrations=True, output_root=tmp_path)
        rep = check_training_readiness(data_dir=tmp_path, check_db=False)
        assert rep.trl_available is True
        assert rep.peft_available is True
        assert rep.datasets_available is True
        assert rep.dataset_manifest_valid is True
        assert rep.ready_for_dry_run is True

    def test_trl_peft_sft_trainer_tiny_cpu_smoke_with_premasked_labels(self, tmp_path: Path):
        """
        Real offline TRL SFTTrainer + PEFT LoraConfig forward/backward step using a tiny
        synthetic LlamaConfig + genuine HuggingFace PreTrainedTokenizerFast (no HF downloads)
        proving loss_type='nll' + pretokenized -100 completion labels execute cleanly.
        """
        import json
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Split
        from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
        from trl import SFTTrainer
        from tests.test_training_dataset import _DeterministicDummyTokenizer

        curator_res = DatasetCurator().build(include_controlled_demonstrations=True, output_root=tmp_path / "data")
        assert curator_res.train_count >= 2

        rows = [
            json.loads(line)
            for line in (tmp_path / "data" / "splits" / "train.jsonl").read_text(encoding="utf-8").splitlines()[:2]
        ]

        specials = [
            "<BOS_TOKEN>",
            "<|START_OF_TURN_TOKEN|>",
            "<|END_OF_TURN_TOKEN|>",
            "<|SYSTEM_TOKEN|>",
            "<|USER_TOKEN|>",
            "<|CHATBOT_TOKEN|>",
            "<|START_RESPONSE|>",
            "<|END_RESPONSE|>",
        ]
        # Build word/punctuation vocabulary from rendered curated rows so sequences fit cleanly <= 2048 tokens
        import re
        from training.data.formatter import render_tiny_aya_chat

        vocab: dict[str, int] = {"<pad>": 0, "<eos>": 1, "<unk>": 2}
        for sp in specials:
            vocab[sp] = len(vocab)
        for r in rows:
            rendered = render_tiny_aya_chat(r["messages"])
            for tok_str in re.findall(r"[A-Za-z0-9_]+|\S|\s+", rendered):
                if tok_str not in vocab:
                    vocab[tok_str] = len(vocab)

        backend = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
        backend.pre_tokenizer = Split(r"[A-Za-z0-9_]+|\S|\s+", "isolated")
        backend.add_special_tokens(specials)
        tok = PreTrainedTokenizerFast(
            tokenizer_object=backend,
            pad_token="<pad>",
            eos_token="<eos>",
            unk_token="<unk>",
            additional_special_tokens=specials,
            chat_template=_DeterministicDummyTokenizer.chat_template,
        )

        factory = DBMedicSFTTrainerFactory()
        ds = factory._prepare_hf_dataset(rows, tok, max_seq_length=2048)
        assert "input_ids" in ds.column_names and "labels" in ds.column_names
        assert -100 in ds[0]["labels"] and any(x != -100 for x in ds[0]["labels"])

        tiny_cfg = LlamaConfig(
            vocab_size=max(512, len(tok) + 8),
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
        tiny_model = LlamaForCausalLM(tiny_cfg)
        sft_yaml = yaml.safe_load((PROJECT_ROOT / "training/configs/sft_config.yaml").read_text(encoding="utf-8"))
        lora = build_lora_config(sft_yaml)
        trl_args = build_sft_config(
            sft_yaml,
            output_dir=tmp_path / "smoke_out",
            has_eval_dataset=False,
            cuda_available=False,
            overrides={"max_steps": 1, "num_train_epochs": 1, "save_strategy": "no"},
        )

        trainer = SFTTrainer(
            model=tiny_model,
            args=trl_args,
            train_dataset=ds,
            processing_class=tok,
            peft_config=lora,
        )
        out = trainer.train()
        assert out.training_loss >= 0.0


@pytest.mark.model
class TestLiveModelReadiness:
    def test_live_hf_token_present_for_tiny_aya_sft(self):
        rep = check_training_readiness(check_db=False)
        if not rep.hf_token_present or not rep.cuda_available:
            pytest.skip("HF_TOKEN and CUDA GPU required for live CohereLabs/tiny-aya-global run.")
        assert rep.ready_for_live_training is True
