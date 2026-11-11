"""
training/agenttune/
===================
AgentTune integration layer for DBMedic Phase 4.

Adapts DBMedic diagnostic trajectories, read-only Phase 2 tools, failures,
and repairs to AgentTune's native interfaces (`EventLog`, `Project`,
`DictToolHarness`, `Trajectory`, `SelfHealLoop`, `create_distill_trainer`,
`create_agentic_trainer`).
"""

from training.agenttune.distillation import (
    DBMedicSFTTrainerFactory,
    ReadinessReport,
    check_training_readiness,
    run_sft_distillation,
)
from training.agenttune.grpo_training import (
    GRPOGatingReport,
    build_readonly_grpo_tools,
    check_grpo_gating,
    run_grpo_experiment,
)
from training.agenttune.trainer import DBMedicTrainingPipeline
from training.agenttune.trajectory_adapter import (
    build_dbmedic_readonly_harness,
    event_log_to_dbmedic_messages,
    to_agenttune_event_log,
    to_agenttune_failure,
    to_agenttune_project,
    to_agenttune_training_example,
    to_agenttune_trajectory,
)

__all__ = [
    "DBMedicSFTTrainerFactory",
    "DBMedicTrainingPipeline",
    "GRPOGatingReport",
    "ReadinessReport",
    "build_dbmedic_readonly_harness",
    "build_readonly_grpo_tools",
    "check_grpo_gating",
    "check_training_readiness",
    "event_log_to_dbmedic_messages",
    "run_grpo_experiment",
    "run_sft_distillation",
    "to_agenttune_event_log",
    "to_agenttune_failure",
    "to_agenttune_project",
    "to_agenttune_training_example",
    "to_agenttune_trajectory",
]

