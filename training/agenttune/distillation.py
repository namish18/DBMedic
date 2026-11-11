"""
training/agenttune/distillation.py
==================================
Primary Phase 4 AgentTune SFT distillation pipeline for CohereLabs/tiny-aya-global.

Integrates DBMedic's verified curated dataset & strict JSON alternation formatter
with AgentTune's native `Project.distill()` lifecycle, TRL `SFTTrainer`, and PEFT LoRA,
while enforcing preflight hardware/auth checks and strict zero-fabrication reporting.
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from agenttune.agentic.project import Project
from agenttune.core.backend_factory import TRL_AVAILABLE
from agenttune.utils.provenance import write_provenance
from training.agenttune.trajectory_adapter import to_agenttune_project
from training.data.formatter import (
    enforce_strict_alternation,
    tokenize_and_mask_sft_row,
)
from training.data.validator import validate_dataset_artifacts

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_SFT_CONFIG_PATH = PROJECT_ROOT / "training" / "configs" / "sft_config.yaml"

load_dotenv(PROJECT_ROOT / ".env", override=False)


def _load_yaml(path: str | Path | None = None) -> dict[str, Any]:
    target = Path(path) if path else DEFAULT_SFT_CONFIG_PATH
    with target.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _detect_hf_token() -> tuple[bool, str | None]:
    for key in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        val = os.getenv(key)
        if val and val.strip():
            return True, key
    return False, None


def _get_hf_token_secret() -> str | None:
    """Return HF token strictly in memory; NEVER log or serialize this string."""
    for key in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        val = os.getenv(key)
        if val and val.strip():
            return val.strip()
    return None


def _probe_db_available(timeout_sec: int = 1) -> bool:
    """Safely check whether dbmedic_lab PostgreSQL is currently reachable."""
    try:
        from agent.tools import get_lab_connection

        with get_lab_connection(read_only=True, timeout_sec=timeout_sec) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
                cur.fetchone()
        return True
    except Exception:
        return False


def _probe_hardware() -> dict[str, Any]:
    cuda_available = False
    gpu_name: str | None = None
    gpu_vram_gb = 0.0
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            props = torch.cuda.get_device_properties(0)
            gpu_name = str(props.name)
            gpu_vram_gb = round(props.total_memory / (1024**3), 2)
    except Exception:
        pass

    free_ram_gb = 0.0
    total_ram_gb = 0.0
    try:
        import psutil

        vm = psutil.virtual_memory()
        free_ram_gb = round(vm.available / (1024**3), 2)
        total_ram_gb = round(vm.total / (1024**3), 2)
    except Exception:
        pass

    return {
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "gpu_vram_gb": gpu_vram_gb,
        "free_ram_gb": free_ram_gb,
        "total_ram_gb": total_ram_gb,
    }


@dataclass
class ReadinessReport:
    """Honest preflight readiness summary for DBMedic AgentTune SFT training."""

    model_id: str
    hf_token_present: bool
    hf_token_env_var: str | None
    cuda_available: bool
    gpu_name: str | None
    gpu_vram_gb: float
    free_ram_gb: float
    total_ram_gb: float
    db_available: bool
    trl_available: bool
    peft_available: bool
    datasets_available: bool
    agenttune_available: bool
    train_dataset_exists: bool
    validation_dataset_exists: bool
    dataset_manifest_valid: bool
    ready_for_dry_run: bool
    ready_for_live_training: bool
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checked_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def check_training_readiness(
    *,
    config_path: str | Path | None = None,
    data_dir: str | Path | None = None,
    model_id_override: str | None = None,
    allow_cpu_smoke: bool = False,
    check_db: bool = True,
) -> ReadinessReport:
    """
    Verify software packages, dataset curation integrity, HF authentication,
    GPU/RAM capacity, and DB availability before launching live SFT.
    """
    cfg = _load_yaml(config_path)
    model_cfg = cfg.get("model", {})
    hw_gate = cfg.get("hardware_gate", {})

    model_id = str(model_id_override or model_cfg.get("base_model_id", "CohereLabs/tiny-aya-global"))
    is_local_model = Path(model_id).exists()

    hf_ok, hf_var = _detect_hf_token()
    hw = _probe_hardware()
    db_ok = _probe_db_available() if check_db else False

    peft_ok = False
    try:
        import peft  # noqa: F401

        peft_ok = True
    except Exception:
        pass

    datasets_ok = False
    try:
        import datasets  # noqa: F401

        datasets_ok = True
    except Exception:
        pass

    data_root = Path(data_dir) if data_dir else PROJECT_ROOT / "training" / "data"
    train_exists = (data_root / "splits" / "train.jsonl").exists()
    val_exists = (data_root / "splits" / "validation.jsonl").exists()
    manifest_valid = False
    if train_exists and val_exists and (data_root / "dataset_manifest.json").exists():
        val_rep = validate_dataset_artifacts(data_root)
        manifest_valid = val_rep.passed
    else:
        val_rep = None

    blockers: list[str] = []
    warnings: list[str] = []

    if not TRL_AVAILABLE:
        blockers.append("TRL agentic backend is unavailable in AgentTune.")
    if not peft_ok:
        blockers.append("peft package is not installed.")
    if not datasets_ok:
        blockers.append("datasets package is not installed.")
    if not train_exists or not val_exists:
        blockers.append(
            "Curated training splits missing; run python scripts/build_training_dataset.py first."
        )
    elif not manifest_valid:
        errs = val_rep.errors if val_rep else ["Dataset validation failed."]
        blockers.append(f"Dataset validation failed: {errs[0]}")

    require_hf = bool(model_cfg.get("require_hf_token", True)) and not is_local_model
    if require_hf and not hf_ok:
        blockers.append(
            f"HF_TOKEN / HUGGINGFACE_HUB_TOKEN not found in environment/.env for gated model '{model_id}'."
        )

    require_cuda = bool(hw_gate.get("require_cuda_for_full_run", True)) and not allow_cpu_smoke
    min_ram_gb = float(hw_gate.get("min_free_ram_gb", 6.0))
    if require_cuda and not hw["cuda_available"]:
        blockers.append(
            "No CUDA GPU detected (torch.cuda.is_available() == False). Full Tiny Aya SFT requires >=6-8 GB GPU VRAM."
        )
    elif not hw["cuda_available"] and hw["free_ram_gb"] > 0 and hw["free_ram_gb"] < min_ram_gb:
        blockers.append(
            f"Insufficient free RAM ({hw['free_ram_gb']} GB < {min_ram_gb} GB) for CPU execution of '{model_id}'."
        )

    if not db_ok:
        warnings.append(
            "dbmedic_lab PostgreSQL is currently unreachable offline; offline curated dataset SFT preflight remains valid, but live trajectory collection and DB evaluation are blocked."
        )

    dry_run_ok = train_exists and val_exists and manifest_valid and TRL_AVAILABLE and peft_ok and datasets_ok
    live_ok = len(blockers) == 0

    return ReadinessReport(
        model_id=model_id,
        hf_token_present=hf_ok,
        hf_token_env_var=hf_var,
        cuda_available=hw["cuda_available"],
        gpu_name=hw["gpu_name"],
        gpu_vram_gb=hw["gpu_vram_gb"],
        free_ram_gb=hw["free_ram_gb"],
        total_ram_gb=hw["total_ram_gb"],
        db_available=db_ok,
        trl_available=bool(TRL_AVAILABLE),
        peft_available=peft_ok,
        datasets_available=datasets_ok,
        agenttune_available=True,
        train_dataset_exists=train_exists,
        validation_dataset_exists=val_exists,
        dataset_manifest_valid=manifest_valid,
        ready_for_dry_run=dry_run_ok,
        ready_for_live_training=live_ok,
        blockers=blockers,
        warnings=warnings,
    )


def _normalize_agenttune_or_curated_row(row: dict[str, Any]) -> list[dict[str, str]]:
    """
    Normalize a row produced either by DBMedic curated JSONL OR by AgentTune's
    `EventLog.as_dataset_rows("sft")` into Tiny Aya strict-alternation DBMedic JSON messages.
    """
    normalized_in: list[dict[str, Any]] = []
    for m in row.get("messages") or []:
        role = str(m.get("role", "user")).lower().strip()
        content = str(m.get("content", "") or "")
        # Unwrap AgentTune's <tool_call>{name, arguments}</tool_call> back to DBMedic JSON protocol
        if role == "assistant" and content.startswith("<tool_call>") and content.endswith("</tool_call>"):
            inner = content[len("<tool_call>") : -len("</tool_call>")].strip()
            try:
                parsed = json.loads(inner)
                if isinstance(parsed, dict) and "name" in parsed and parsed.get("type") != "tool_call":
                    content = json.dumps(
                        {
                            "type": "tool_call",
                            "tool": parsed.get("name"),
                            "arguments": parsed.get("arguments", {}),
                            "reasoning": "Execute read-only PostgreSQL diagnostic check.",
                        },
                        ensure_ascii=False,
                    )
            except json.JSONDecodeError:
                pass
        normalized_in.append({**m, "role": role, "content": content})

    return enforce_strict_alternation(normalized_in, fold_tool_outputs_into_user=True)


def build_lora_config(sft_cfg: dict[str, Any]) -> Any:
    """
    Construct a PEFT `LoraConfig` matching `sft_config.yaml`, safely respecting
    installed `peft` kwargs and preventing `chunked_nll` + `lm_head` clashes.
    """
    from peft import LoraConfig

    peft_cfg = sft_cfg.get("peft", {})
    kwargs: dict[str, Any] = {
        "r": int(peft_cfg.get("r", 16)),
        "lora_alpha": int(peft_cfg.get("lora_alpha", 32)),
        "lora_dropout": float(peft_cfg.get("lora_dropout", 0.05)),
        "bias": str(peft_cfg.get("bias", "none")),
        "task_type": str(peft_cfg.get("task_type", "CAUSAL_LM")),
        "target_modules": peft_cfg.get("target_modules", "all-linear"),
    }
    sig_params = inspect.signature(LoraConfig.__init__).parameters
    exclude_mods = peft_cfg.get("exclude_modules")
    if exclude_mods and "exclude_modules" in sig_params:
        kwargs["exclude_modules"] = list(exclude_mods)
    return LoraConfig(**kwargs)


def build_sft_config(
    sft_cfg: dict[str, Any],
    *,
    output_dir: str | Path,
    has_eval_dataset: bool = True,
    cuda_available: bool = False,
    overrides: dict[str, Any] | None = None,
) -> Any:
    """
    Build a `trl.SFTConfig` compatible with TRL 1.7.1 pretokenized completion-only labels.
    Notice:
      - Sets `loss_type="nll"` to avoid TRL's chunked_nll PEFT `lm_head` conflict.
      - Sets `completion_only_loss=False` at TRL collation level because DBMedic
        pre-masks `labels` with `-100` for all system/user/tool turns cleanly even when
        Tiny Aya's Jinja template lacks `{% generation %}` blocks.
    """
    from trl import SFTConfig

    tr_cfg = {**sft_cfg.get("training", {}), **(overrides or {})}
    sig_params = inspect.signature(SFTConfig.__init__).parameters

    use_bf16 = bool(tr_cfg.get("bf16", True)) and cuda_available
    use_fp16 = bool(tr_cfg.get("fp16", False)) and cuda_available and not use_bf16

    candidate: dict[str, Any] = {
        "output_dir": str(output_dir),
        "seed": int(sft_cfg.get("experiment", {}).get("seed", 42)),
        "num_train_epochs": float(tr_cfg.get("num_train_epochs", 3)),
        "learning_rate": float(tr_cfg.get("learning_rate", 2.0e-4)),
        "lr_scheduler_type": str(tr_cfg.get("lr_scheduler_type", "cosine")),
        "warmup_ratio": float(tr_cfg.get("warmup_ratio", 0.1)),
        "per_device_train_batch_size": int(tr_cfg.get("per_device_train_batch_size", 1)),
        "per_device_eval_batch_size": int(tr_cfg.get("per_device_eval_batch_size", 1)),
        "gradient_accumulation_steps": int(tr_cfg.get("gradient_accumulation_steps", 4)),
        "max_length": int(tr_cfg.get("max_length", 2048)),
        "loss_type": str(tr_cfg.get("loss_type", "nll")),
        "completion_only_loss": False,  # Pre-tokenized labels carry explicit -100 completion mask
        "packing": bool(tr_cfg.get("packing", False)),
        "gradient_checkpointing": bool(tr_cfg.get("gradient_checkpointing", False)) and cuda_available,
        "bf16": use_bf16,
        "fp16": use_fp16,
        "logging_steps": int(tr_cfg.get("logging_steps", 1)),
        "eval_strategy": str(tr_cfg.get("eval_strategy", "epoch")) if has_eval_dataset else "no",
        "save_strategy": str(tr_cfg.get("save_strategy", "epoch")),
        "save_total_limit": int(tr_cfg.get("save_total_limit", 2)),
        "report_to": tr_cfg.get("report_to", "none"),
    }
    filtered = {k: v for k, v in candidate.items() if k in sig_params}
    return SFTConfig(**filtered)


class DBMedicSFTTrainerFactory:
    """
    AgentTune `trainer_factory` callable plugged into `Project.distill()`.

    Handles:
      - Secure HF token lookup (without logging)
      - Tiny Aya strict user/assistant alternation + folded tool observations
      - Token-level completion-only label masking (`-100` on non-assistant turns)
      - LoRA PEFT wrapping (`target_modules="all-linear"`, `exclude_modules=["lm_head"]`)
      - Safe `SFTTrainer` construction for TRL 1.7.1
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
        *,
        eval_rows: list[dict[str, Any]] | None = None,
        model_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.cfg = _load_yaml(config_path)
        self.eval_rows = eval_rows or []
        self.model_kwargs = model_kwargs or {}
        self.model = None
        self.tokenizer = None
        self.sft_trainer = None

    def _prepare_hf_dataset(
        self,
        rows: list[dict[str, Any]],
        tokenizer: Any,
        max_seq_length: int,
    ) -> Any:
        from datasets import Dataset

        tokenized_rows: list[dict[str, Any]] = []
        for row in rows:
            strict_msgs = _normalize_agenttune_or_curated_row(row)
            tok = tokenize_and_mask_sft_row(
                strict_msgs,
                tokenizer,
                max_seq_length=max_seq_length,
            )
            if not any(lbl != -100 for lbl in tok["labels"]):
                raise ValueError(
                    f"SFT row left zero unmasked assistant tokens after max_seq_length={max_seq_length} truncation; "
                    "increase training.max_length to avoid NaN cross-entropy loss."
                )
            tokenized_rows.append(
                {
                    "input_ids": tok["input_ids"],
                    "attention_mask": tok["attention_mask"],
                    "labels": tok["labels"],
                }
            )
        return Dataset.from_list(tokenized_rows)

    def __call__(
        self,
        *,
        model: str,
        train_dataset: list[dict[str, Any]],
        eval_dataset: list[dict[str, Any]] | None = None,
        **trainer_kwargs: Any,
    ) -> Any:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import SFTTrainer

        hf_token = _get_hf_token_secret()
        model_cfg = self.cfg.get("model", {})
        trust_remote = bool(model_cfg.get("trust_remote_code", False))
        cuda_ok = bool(torch.cuda.is_available())
        dtype = torch.bfloat16 if cuda_ok else torch.float32
        device_map = model_cfg.get("device_map", "auto") if cuda_ok else None

        tok_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote}
        if hf_token and not Path(str(model)).exists():
            tok_kwargs["token"] = hf_token

        self.tokenizer = AutoTokenizer.from_pretrained(model, **tok_kwargs)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "trust_remote_code": trust_remote,
            **self.model_kwargs,
        }
        if device_map is not None:
            load_kwargs["device_map"] = device_map
        if hf_token and not Path(str(model)).exists():
            load_kwargs["token"] = hf_token

        self.model = AutoModelForCausalLM.from_pretrained(model, **load_kwargs)

        max_len = int(self.cfg.get("training", {}).get("max_length", 2048))
        raw_eval = eval_dataset if eval_dataset is not None else self.eval_rows

        train_ds = self._prepare_hf_dataset(list(train_dataset), self.tokenizer, max_len)
        eval_ds = (
            self._prepare_hf_dataset(list(raw_eval), self.tokenizer, max_len)
            if raw_eval
            else None
        )

        out_dir = trainer_kwargs.pop("output_dir", self.cfg.get("experiment", {}).get("output_dir", "training/experiments/sft"))
        sft_args = build_sft_config(
            self.cfg,
            output_dir=out_dir,
            has_eval_dataset=eval_ds is not None,
            cuda_available=cuda_ok,
            overrides=trainer_kwargs,
        )

        peft_cfg = build_lora_config(self.cfg) if self.cfg.get("peft", {}).get("use_peft", True) else None

        self.sft_trainer = SFTTrainer(
            model=self.model,
            args=sft_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            processing_class=self.tokenizer,
            peft_config=peft_cfg,
        )
        return self.sft_trainer


def _load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _compute_dir_size_mb(path: Path) -> float | None:
    if not path.exists():
        return None
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return round(total / (1024 * 1024), 4)


def run_sft_distillation(
    *,
    config_path: str | Path | None = None,
    data_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    model_id: str | None = None,
    dry_run: bool = False,
    allow_cpu_smoke: bool = False,
    check_db: bool = True,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Execute or validate DBMedic AgentTune SFT distillation.

    When `dry_run=True`, verifies curated data, AgentTune `Project.distill` projection,
    LoRA config assembly, and writes configuration/provenance snapshots cleanly without
    fabricating any training metrics or weights.

    When `dry_run=False` and `readiness.ready_for_live_training` is False, writes an
    honest `status="blocked"` artifact explaining every blocker and returns without
    fabricating loss curves or checkpoints.
    """
    cfg = _load_yaml(config_path)
    out_path = Path(output_dir) if output_dir else PROJECT_ROOT / cfg.get("experiment", {}).get("output_dir", "training/experiments/sft")
    out_path.mkdir(parents=True, exist_ok=True)

    data_root = Path(data_dir) if data_dir else PROJECT_ROOT / "training" / "data"
    resolved_model = str(model_id or cfg.get("model", {}).get("base_model_id", "CohereLabs/tiny-aya-global"))

    readiness = check_training_readiness(
        config_path=config_path,
        data_dir=data_root,
        model_id_override=resolved_model,
        allow_cpu_smoke=allow_cpu_smoke,
        check_db=check_db,
    )

    train_rows = _load_jsonl_rows(data_root / "splits" / "train.jsonl")
    val_rows = _load_jsonl_rows(data_root / "splits" / "validation.jsonl")
    manifest_path = data_root / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}

    # Save reproducible experiment configuration + dataset manifest snapshot + provenance
    config_snapshot = dict(cfg)
    config_snapshot["resolved_model_id"] = resolved_model
    (out_path / "training_config.yaml").write_text(
        yaml.safe_dump(config_snapshot, sort_keys=False),
        encoding="utf-8",
    )
    (out_path / "dataset_manifest_snapshot.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    write_provenance(
        out_path,
        method="agenttune.sft_distill",
        base_model=resolved_model,
        dataset=str(data_root),
        params={
            "lora": cfg.get("peft", {}),
            "training": cfg.get("training", {}),
            "train_examples": len(train_rows),
            "validation_examples": len(val_rows),
            "dry_run": dry_run,
        },
    )

    # Preview AgentTune Project projection (100% GPU-free)
    project: Project = to_agenttune_project(train_rows, tier="full")
    agenttune_sft_preview = project.sft_dataset("sft") if train_rows else []
    lora_cfg = build_lora_config(cfg)

    base_telemetry = {
        "training_time_sec": None,
        "gpu_memory_usage_mb": None,
        "inference_latency_ms": None,
        "adapter_size_mb": None,
        "dataset_size": len(train_rows),
        "validation_size": len(val_rows),
        "training_loss": None,
    }

    if dry_run:
        status_payload = {
            "schema_version": "dbmedic.sft_status/1",
            "status": "dry_run_validated" if readiness.ready_for_dry_run else "dry_run_failed",
            "executed_training": False,
            "model_id": resolved_model,
            "agenttune_project_trajectories": len(project.trajectories),
            "agenttune_sft_rows": len(agenttune_sft_preview),
            "lora_config": {
                "r": lora_cfg.r,
                "lora_alpha": lora_cfg.lora_alpha,
                "lora_dropout": lora_cfg.lora_dropout,
                "target_modules": str(lora_cfg.target_modules),
                "exclude_modules": list(getattr(lora_cfg, "exclude_modules", []) or []),
            },
            "readiness": readiness.to_dict(),
            "telemetry": base_telemetry,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "note": "Preflight dry-run only; no model weights downloaded or fine-tuned.",
        }
        (out_path / "sft_status.json").write_text(json.dumps(status_payload, indent=2), encoding="utf-8")
        return status_payload

    if not readiness.ready_for_live_training:
        status_payload = {
            "schema_version": "dbmedic.sft_status/1",
            "status": "blocked",
            "executed_training": False,
            "model_id": resolved_model,
            "agenttune_project_trajectories": len(project.trajectories),
            "agenttune_sft_rows": len(agenttune_sft_preview),
            "readiness": readiness.to_dict(),
            "blockers": readiness.blockers,
            "telemetry": base_telemetry,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "note": (
                "Live SFT training could not execute in current environment due to explicit blockers "
                "(no fabricated training metrics or checkpoints written)."
            ),
        }
        (out_path / "sft_status.json").write_text(json.dumps(status_payload, indent=2), encoding="utf-8")
        return status_payload

    # Live AgentTune Project.distill() execution (only reached when auth, hardware & data pass!)
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    factory = DBMedicSFTTrainerFactory(config_path=config_path, eval_rows=val_rows)
    start_ts = time.perf_counter()
    trainer_kw = {"output_dir": str(out_path), **(overrides or {})}
    train_output = project.distill(
        resolved_model,
        trainer_factory=factory,
        fmt="sft",
        **trainer_kw,
    )
    elapsed_sec = round(time.perf_counter() - start_ts, 2)

    if factory.sft_trainer is not None:
        factory.sft_trainer.save_model(str(out_path))
    if factory.tokenizer is not None:
        factory.tokenizer.save_pretrained(str(out_path))

    peak_vram_mb = (
        round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)
        if torch.cuda.is_available()
        else None
    )
    train_loss = getattr(train_output, "training_loss", None)
    if train_loss is None and isinstance(train_output, dict):
        train_loss = train_output.get("training_loss") or train_output.get("train_loss")

    status_payload = {
        "schema_version": "dbmedic.sft_status/1",
        "status": "completed",
        "executed_training": True,
        "model_id": resolved_model,
        "output_dir": str(out_path),
        "agenttune_project_trajectories": len(project.trajectories),
        "agenttune_sft_rows": len(agenttune_sft_preview),
        "readiness": readiness.to_dict(),
        "telemetry": {
            "training_time_sec": elapsed_sec,
            "gpu_memory_usage_mb": peak_vram_mb,
            "inference_latency_ms": None,
            "adapter_size_mb": _compute_dir_size_mb(out_path),
            "dataset_size": len(train_rows),
            "validation_size": len(val_rows),
            "training_loss": float(train_loss) if train_loss is not None else None,
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_path / "sft_status.json").write_text(json.dumps(status_payload, indent=2), encoding="utf-8")
    return status_payload
