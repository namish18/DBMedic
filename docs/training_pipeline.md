# DBMedic Phase 4 Training Pipeline

DBMedic uses a disciplined two-stage fine-tuning strategy powered by AgentTune, starting with offline-curated Supervised Fine-Tuning (SFT) distillation as the primary production path and gating Agentic GRPO behind strict prerequisites.

## Primary Path: AgentTune LoRA SFT Distillation (`training/agenttune/distillation.py`)

- **Model**: `CohereLabs/tiny-aya-global` (`cohere2` family)
- **Method**: AgentTune `Project.distill(..., trainer_factory=DBMedicSFTTrainerFactory)` backed by TRL `SFTTrainer` + PEFT `LoraConfig`
- **LoRA Configuration (`training/configs/sft_config.yaml`)**:
  - `r: 16`, `lora_alpha: 32`, `lora_dropout: 0.05`, `bias: none`, `task_type: CAUSAL_LM`
  - `target_modules: all-linear` with `exclude_modules: ["lm_head"]` (explicit fallback list retained)
  - `loss_type: nll`, `learning_rate: 2.0e-4`, `lr_scheduler_type: cosine`, `warmup_ratio: 0.1`, `num_train_epochs: 3`, `max_length: 2048`
- **Loss Masking**: Token-level `-100` masking on system prompts, user prompts, and tool outputs; gradient loss is computed ONLY on assistant JSON `tool_call` and `final_answer` turns.

### Readiness & Preflight Checks (`check_training_readiness`)
Before any live weight download or optimization runs, `DBMedicTrainingPipeline.preflight()` verifies:
1. Validated curated splits (`train.jsonl`, `validation.jsonl`) and SHA-256 `dataset_manifest.json`.
2. `TRL_AVAILABLE == True`, `peft`, `datasets`, `agenttune` installed with compatible NumPy (`1.26.4`) / PyArrow (`23.0.1`) bindings.
3. Presence of `HF_TOKEN` / `HUGGINGFACE_HUB_TOKEN` in environment/`.env` (token values are never logged).
4. GPU/CUDA availability (`>=6 GB VRAM`) or explicit `--allow-cpu-smoke` for local tiny test models only.
5. `dbmedic_lab` PostgreSQL status (warning only for offline SFT curation; mandatory blocker for live trajectory collection / GRPO).

If prerequisites are unmet during non-dry-run invocation, `sft_status.json` records `"status": "blocked"` alongside exact blockers and `null` telemetry — never fabricating training runs or checkpoints.

## Secondary / Experimental Path: Gated Agentic GRPO (`training/agenttune/grpo_training.py`)

GRPO uses `create_agentic_trainer("grpo", backend="trl", ...)` with:
- Strictly read-only Phase 2 tools (`DBMedicReadOnlyAgentTuneTool` wrapping `ToolRegistry` + `ToolValidator`)
- Cohere tool prompt fallback (`tools_fallback_prompt: cohere`) and user-folded tool observations
- Composite deterministic reward function (`make_agenttune_reward_fn`)

### Mandatory GRPO Gating (`check_grpo_gating`)
GRPO is automatically blocked (`"status": "gated_blocked"`) unless ALL prerequisites pass:
1. Saved SFT adapter/model checkpoint exists in `training/experiments/sft/`
2. `HF_TOKEN` configured
3. CUDA GPU available with `>= 12.0 GB` VRAM
4. Live PostgreSQL `dbmedic_lab` connection healthy (`SELECT 1`)
5. Read-only tool safety verification passes (zero mutating/SQL tools)

## CLI Reproduction Commands

```powershell
python scripts/collect_training_trajectories.py --generate-controlled-failures-sample
python scripts/analyze_failures.py
python scripts/build_training_dataset.py
python scripts/validate_training_dataset.py
python scripts/train_agent.py --mode preflight
python scripts/train_agent.py --mode sft --dry-run
python scripts/train_agent.py --mode grpo --dry-run
python scripts/evaluate_trained_agent.py --split validation --dry-run
python scripts/compare_models.py --dry-run
```

