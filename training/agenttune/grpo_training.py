"""
training/agenttune/grpo_training.py
===================================
Secondary/experimental Phase 4 AgentTune agentic GRPO pipeline.

GRPO is STRICTLY gated behind verified SFT distillation, CUDA hardware (>=12 GB VRAM),
HF authentication, and live `dbmedic_lab` PostgreSQL connectivity. Exposes ONLY the
8 read-only diagnostic tools (`APPROVED_DIAGNOSTIC_TOOLS`); mutating tools such as
`apply_remediation` and arbitrary SQL execution are hard-excluded.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from agent.core.tool_registry import ToolRegistry
from agent.core.tool_validator import ToolValidator
from agenttune.agentic.tools.base import BaseTool, ToolResult
from agenttune.core.backend_factory import TRL_AVAILABLE, create_agentic_trainer
from agenttune.utils.provenance import write_provenance
from training.agenttune.distillation import (
    _detect_hf_token,
    _probe_db_available,
    _probe_hardware,
)
from training.failure_analysis.classifier import APPROVED_DIAGNOSTIC_TOOLS
from training.rewards.diagnostic_rewards import make_agenttune_reward_fn

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_GRPO_CONFIG_PATH = PROJECT_ROOT / "training" / "configs" / "grpo_config.yaml"

_DISALLOWED_GRPO_TOOLS = frozenset(
    {
        "apply_remediation",
        "execute_sql",
        "run_sql",
        "create_index",
        "drop_index",
        "drop_table",
        "alter_table",
    }
)


def _load_grpo_config(config_path: str | Path | None = None) -> dict[str, Any]:
    target = Path(config_path) if config_path else DEFAULT_GRPO_CONFIG_PATH
    with target.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _has_saved_checkpoint(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    marker_files = (
        "adapter_config.json",
        "adapter_model.safetensors",
        "adapter_model.bin",
        "config.json",
        "model.safetensors",
        "pytorch_model.bin",
    )
    return any((path / m).exists() for m in marker_files)


class DBMedicReadOnlyAgentTuneTool(BaseTool):
    """AgentTune BaseTool wrapper enforcing ToolRegistry + ToolValidator read-only safety."""

    def __init__(
        self,
        *,
        name: str,
        description: str,
        parameters_schema: dict[str, Any],
        required: list[str],
        registry: ToolRegistry,
        validator: ToolValidator,
    ) -> None:
        if name not in APPROVED_DIAGNOSTIC_TOOLS or name in _DISALLOWED_GRPO_TOOLS:
            raise PermissionError(f"Tool '{name}' is forbidden in GRPO read-only environment.")
        self.name = name
        self.description = description
        self._schema_props = parameters_schema
        self._required = list(required)
        self._registry = registry
        self._validator = validator

    def _parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": self._schema_props,
            "required": self._required,
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        res = self._validator.validate(self.name, kwargs)
        if not res.is_valid:
            return ToolResult(success=False, output=None, error=res.error)
        try:
            out = self._registry.execute(self.name, res.sanitized_arguments)
            return ToolResult(success=True, output=out)
        except Exception as exc:
            return ToolResult(success=False, output=None, error=str(exc))


def build_readonly_grpo_tools(
    *,
    allowed_tools: list[str] | None = None,
) -> list[DBMedicReadOnlyAgentTuneTool]:
    """
    Build strictly read-only AgentTune tools backed by Phase 2 ToolRegistry.
    Never exposes mutating tools or arbitrary SQL interfaces.
    """
    registry = ToolRegistry()
    registry.initialize()
    validator = ToolValidator(registry=registry, max_calls_per_session=10)

    requested = allowed_tools or sorted(APPROVED_DIAGNOSTIC_TOOLS)
    tools: list[DBMedicReadOnlyAgentTuneTool] = []
    for name in requested:
        if name in _DISALLOWED_GRPO_TOOLS or name not in APPROVED_DIAGNOSTIC_TOOLS:
            raise PermissionError(
                f"Refusing to expose unsafe or unapproved tool '{name}' in GRPO environment."
            )
        if not registry.has_tool(name):
            raise KeyError(f"Approved diagnostic tool '{name}' missing from ToolRegistry.")
        spec = registry.get_schema(name)
        tools.append(
            DBMedicReadOnlyAgentTuneTool(
                name=name,
                description=spec["description"],
                parameters_schema=spec["parameters"],
                required=spec["required"],
                registry=registry,
                validator=validator,
            )
        )
    return tools


@dataclass
class GRPOGatingReport:
    """Explicit prerequisite check for experimental GRPO agentic reinforcement learning."""

    passed: bool
    sft_checkpoint_exists: bool
    sft_checkpoint_path: str
    hf_token_present: bool
    cuda_available: bool
    gpu_vram_gb: float
    min_required_vram_gb: float
    live_db_available: bool
    trl_available: bool
    read_only_tools_verified: bool
    num_readonly_tools: int
    blockers: list[str] = field(default_factory=list)
    checked_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def check_grpo_gating(
    *,
    config_path: str | Path | None = None,
    sft_checkpoint_dir: str | Path | None = None,
    check_db: bool = True,
) -> GRPOGatingReport:
    """Verify all mandatory prerequisites before allowing agentic GRPO training."""
    cfg = _load_grpo_config(config_path)
    gate_cfg = cfg.get("gating_requirements", {})
    rollout_cfg = cfg.get("rollout", {})

    ckpt_dir = Path(
        sft_checkpoint_dir
        or gate_cfg.get("sft_checkpoint_dir", "training/experiments/sft")
    )
    if not ckpt_dir.is_absolute():
        ckpt_dir = PROJECT_ROOT / ckpt_dir

    sft_ok = _has_saved_checkpoint(ckpt_dir)
    hf_ok, _ = _detect_hf_token()
    hw = _probe_hardware()
    db_ok = _probe_db_available() if check_db else False
    min_vram = float(gate_cfg.get("min_free_vram_gb", 12.0))

    tools_ok = False
    num_tools = 0
    try:
        tools = build_readonly_grpo_tools(allowed_tools=rollout_cfg.get("allowed_tools"))
        num_tools = len(tools)
        tool_names = {t.name for t in tools}
        tools_ok = tool_names.issubset(APPROVED_DIAGNOSTIC_TOOLS) and not (
            tool_names & _DISALLOWED_GRPO_TOOLS
        )
    except Exception:
        tools_ok = False

    blockers: list[str] = []
    if gate_cfg.get("require_sft_checkpoint", True) and not sft_ok:
        blockers.append(
            f"Missing verified SFT checkpoint at '{ckpt_dir}'; complete SFT distillation before GRPO."
        )
    if gate_cfg.get("require_hf_token", True) and not hf_ok:
        blockers.append("HF_TOKEN / HUGGINGFACE_HUB_TOKEN is not set.")
    if gate_cfg.get("require_cuda_gpu", True) and (
        not hw["cuda_available"] or hw["gpu_vram_gb"] < min_vram
    ):
        blockers.append(
            f"CUDA GPU with >= {min_vram} GB VRAM is required for GRPO rollouts "
            f"(detected cuda={hw['cuda_available']}, vram={hw['gpu_vram_gb']} GB)."
        )
    if gate_cfg.get("require_live_db", True) and not db_ok:
        blockers.append(
            "Live dbmedic_lab PostgreSQL database must be online to run real diagnostic GRPO rollouts."
        )
    if not TRL_AVAILABLE:
        blockers.append("TRL GRPO agentic backend is unavailable.")
    if not tools_ok:
        blockers.append("Read-only GRPO diagnostic tool set failed safety verification.")

    return GRPOGatingReport(
        passed=len(blockers) == 0,
        sft_checkpoint_exists=sft_ok,
        sft_checkpoint_path=str(ckpt_dir),
        hf_token_present=hf_ok,
        cuda_available=hw["cuda_available"],
        gpu_vram_gb=hw["gpu_vram_gb"],
        min_required_vram_gb=min_vram,
        live_db_available=db_ok,
        trl_available=bool(TRL_AVAILABLE),
        read_only_tools_verified=tools_ok,
        num_readonly_tools=num_tools,
        blockers=blockers,
    )


def run_grpo_experiment(
    *,
    config_path: str | Path | None = None,
    data_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    dry_run: bool = False,
    check_db: bool = True,
) -> dict[str, Any]:
    """
    Validate or execute the gated experimental AgentTune GRPO pipeline.
    Never fabricates RL runs when gating prerequisites fail.
    """
    cfg = _load_grpo_config(config_path)
    out_path = Path(output_dir) if output_dir else PROJECT_ROOT / cfg.get("experiment", {}).get("output_dir", "training/experiments/grpo")
    out_path.mkdir(parents=True, exist_ok=True)

    data_root = Path(data_dir) if data_dir else PROJECT_ROOT / "training" / "data"
    rollout_cfg = cfg.get("rollout", {})
    gating = check_grpo_gating(config_path=config_path, check_db=check_db)
    tools = build_readonly_grpo_tools(allowed_tools=rollout_cfg.get("allowed_tools"))
    reward_fn = make_agenttune_reward_fn(
        config_path=PROJECT_ROOT / cfg.get("rewards", {}).get("config_path", "training/rewards/reward_config.yaml")
    )

    (out_path / "grpo_config_snapshot.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8"
    )
    write_provenance(
        out_path,
        method="agenttune.agentic.grpo",
        base_model=str(cfg.get("model", {}).get("base_model_id", "training/experiments/sft")),
        dataset=str(data_root),
        params={
            "gating_passed": gating.passed,
            "num_readonly_tools": len(tools),
            "tools": [t.name for t in tools],
            "dry_run": dry_run,
        },
    )

    if dry_run:
        payload = {
            "schema_version": "dbmedic.grpo_status/1",
            "status": "dry_run_validated" if gating.read_only_tools_verified else "dry_run_failed",
            "executed_training": False,
            "experimental_only": True,
            "readonly_tools": [t.name for t in tools],
            "excluded_tools": list(_DISALLOWED_GRPO_TOOLS),
            "reward_function": reward_fn.__name__,
            "gating": gating.to_dict(),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "note": "GRPO dry-run verified read-only tools, composite reward, and gating rules.",
        }
        (out_path / "grpo_status.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    if not gating.passed:
        payload = {
            "schema_version": "dbmedic.grpo_status/1",
            "status": "gated_blocked",
            "executed_training": False,
            "experimental_only": True,
            "readonly_tools": [t.name for t in tools],
            "excluded_tools": list(_DISALLOWED_GRPO_TOOLS),
            "gating": gating.to_dict(),
            "blockers": gating.blockers,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "note": (
                "Experimental GRPO is blocked by gating checks (SFT checkpoint, CUDA >= 12 GB, "
                "live dbmedic_lab, HF_TOKEN). Zero RL rollouts executed or fabricated."
            ),
        }
        (out_path / "grpo_status.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    # Only executed when ALL 5 strict gates pass
    from datasets import Dataset

    train_path = data_root / "splits" / "train.jsonl"
    prompts = []
    if train_path.exists():
        for line in train_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            prompts.append({"prompt": row.get("query", ""), "answer": row.get("root_cause", "")})

    if not prompts:
        raise RuntimeError(f"No GRPO training prompts found in {train_path}.")

    tr_cfg = cfg.get("training", {})
    peft_cfg = cfg.get("peft", {})
    trainer = create_agentic_trainer(
        algorithm="grpo",
        backend="trl",
        model=gating.sft_checkpoint_path,
        reward_funcs=reward_fn,
        tools=tools,
        train_dataset=Dataset.from_list(prompts),
        output_dir=str(out_path),
        max_steps=int(tr_cfg.get("max_steps", 20)),
        max_completion_length=int(rollout_cfg.get("max_completion_length", 768)),
        num_generations=int(tr_cfg.get("num_generations", 2)),
        per_device_train_batch_size=int(tr_cfg.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(tr_cfg.get("gradient_accumulation_steps", 4)),
        learning_rate=float(tr_cfg.get("learning_rate", 5.0e-6)),
        beta=float(tr_cfg.get("beta", 0.04)),
        temperature=float(tr_cfg.get("temperature", 0.7)),
        bf16=bool(tr_cfg.get("bf16", True)),
        log_completions=bool(tr_cfg.get("log_completions", True)),
        report_to=tr_cfg.get("report_to", "none"),
        peft_config=peft_cfg,
        rollout_backend=rollout_cfg.get("rollout_backend", "transformers"),
        tools_fallback_prompt=rollout_cfg.get("tools_fallback_prompt", "cohere"),
        max_steps_per_turn=int(rollout_cfg.get("max_steps_per_episode", 6)),
    )
    train_res = trainer.train()
    payload = {
        "schema_version": "dbmedic.grpo_status/1",
        "status": "completed",
        "executed_training": True,
        "experimental_only": True,
        "readonly_tools": [t.name for t in tools],
        "gating": gating.to_dict(),
        "train_result": train_res if isinstance(train_res, dict) else {"summary": str(train_res)},
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_path / "grpo_status.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload

