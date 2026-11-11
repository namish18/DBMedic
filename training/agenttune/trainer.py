"""
training/agenttune/trainer.py
=============================
Unified DBMedic Phase 4 AgentTune training orchestrator connecting:
  1. Preflight validation (dataset integrity, hardware, HF auth, live DB)
  2. Primary SFT distillation (`run_sft_distillation`)
  3. Secondary gated GRPO experimentation (`run_grpo_experiment`)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from training.agenttune.distillation import (
    DEFAULT_SFT_CONFIG_PATH,
    ReadinessReport,
    check_training_readiness,
    run_sft_distillation,
)
from training.agenttune.grpo_training import (
    DEFAULT_GRPO_CONFIG_PATH,
    GRPOGatingReport,
    check_grpo_gating,
    run_grpo_experiment,
)
from training.data.validator import DatasetValidationReport, validate_dataset_artifacts


class DBMedicTrainingPipeline:
    """High-level orchestrator for AgentTune SFT and gated GRPO stages."""

    def __init__(
        self,
        *,
        sft_config_path: str | Path = DEFAULT_SFT_CONFIG_PATH,
        grpo_config_path: str | Path = DEFAULT_GRPO_CONFIG_PATH,
        data_dir: str | Path | None = None,
    ) -> None:
        self.sft_config_path = Path(sft_config_path)
        self.grpo_config_path = Path(grpo_config_path)
        self.data_dir = Path(data_dir) if data_dir else None

    def preflight(self, *, check_db: bool = True) -> dict[str, Any]:
        """Run dataset validation, SFT readiness checks, and GRPO gating checks."""
        dataset_report: DatasetValidationReport = validate_dataset_artifacts(self.data_dir)
        sft_readiness: ReadinessReport = check_training_readiness(
            config_path=self.sft_config_path,
            data_dir=self.data_dir,
            check_db=check_db,
        )
        grpo_gating: GRPOGatingReport = check_grpo_gating(
            config_path=self.grpo_config_path,
            check_db=check_db,
        )
        return {
            "dataset_validation": dataset_report.to_dict(),
            "sft_readiness": sft_readiness.to_dict(),
            "grpo_gating": grpo_gating.to_dict(),
        }

    def run_sft(
        self,
        *,
        output_dir: str | Path | None = None,
        model_id: str | None = None,
        dry_run: bool = False,
        allow_cpu_smoke: bool = False,
        check_db: bool = True,
        overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run or validate primary AgentTune SFT distillation."""
        return run_sft_distillation(
            config_path=self.sft_config_path,
            data_dir=self.data_dir,
            output_dir=output_dir,
            model_id=model_id,
            dry_run=dry_run,
            allow_cpu_smoke=allow_cpu_smoke,
            check_db=check_db,
            overrides=overrides,
        )

    def run_grpo(
        self,
        *,
        output_dir: str | Path | None = None,
        dry_run: bool = False,
        check_db: bool = True,
    ) -> dict[str, Any]:
        """Run or validate gated experimental GRPO reinforcement learning."""
        return run_grpo_experiment(
            config_path=self.grpo_config_path,
            data_dir=self.data_dir,
            output_dir=output_dir,
            dry_run=dry_run,
            check_db=check_db,
        )

