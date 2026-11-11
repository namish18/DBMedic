# DBMedic — Phase 4 Technical Report: AgentTune Integration, Failure Learning, and Self-Healing Pipeline

**Project:** DBMedic  
**Phase:** 4 — AgentTune Integration, Failure Learning & Controlled Model Improvement  
**Base Model Target:** `CohereLabs/tiny-aya-global` (`cohere2` family)  
**Primary Training Method:** AgentTune `Project.distill` LoRA SFT Distillation (`trl==1.7.1`, `peft`, `agenttune==1.1.4`)  
**Secondary Experimental Method:** Strictly Gated Agentic GRPO (`create_agentic_trainer("grpo", backend="trl")`)  

---

## 1. Executive Summary & Strict Honesty Statement

Phase 4 transforms DBMedic from a deterministic Phase 2 engine and Phase 3 ReAct agent into an end-to-end **failure-driven self-healing training pipeline** powered by **AgentTune (`v1.1.4`)**. Every pipeline component — 10-category trajectory failure classification, deterministic Phase 2 correction synthesis and verification, Tiny Aya strict-alternation formatting with token-level `-100` completion-only label masking, SHA-256-audited leakage-free dataset splits, YAML-configurable anti-reward-hacking rewards, TRL/PEFT LoRA SFT distillation, gated GRPO experimentation, 8-stage `SelfHealLoop` orchestration, and human-gated A/B model comparison — is implemented, exercised via CLI dry-runs and live local runs, and verified by **160 passing offline unit tests**.

### Strictly Verified vs. Environment-Blocked Boundaries

No results, training loss curves, checkpoints, or model metrics have been fabricated:

- **Completed & Verified Locally (`Exit Code 0`, Reproducible Offline):**
  - **160 / 160 unit tests passing** (`python -m pytest -m "unit" -v`), including a real CPU forward/backward optimization step using `transformers.PreTrainedTokenizerFast`, `LlamaForCausalLM`, `peft.LoraConfig(target_modules="all-linear", exclude_modules=["lm_head"])`, and `trl.SFTTrainer(loss_type="nll")`.
  - Controlled synthetic failure dataset (`training/data/raw/sample_controlled_trajectories.jsonl`, explicitly tagged `"simulated": true`) analyzed across all **10 failure categories** (`19` failure records across `10` trajectories, `1` safety failure).
  - Leakage-free curated dataset built and independently audited (`18` train / `5` validation / `5` answer-stripped `hidden_test` prompts / `10` reserved `baseline` cases excluded, SHA-256 manifest verified).
  - Real preflight readiness, dry-run validation (`sft`, `grpo`, `validation`, `hidden`, `compare_models`), honest live block enforcement (`sft` -> `blocked` exit `2`; `grpo` -> `gated_blocked` exit `2`), and Phase 2 mock diagnostics (`scripts/run_diagnostics.py --mock`).
  - Pre-existing `benchmarks/baseline/results.json` (`[]`) and `summary.json` (`pending`) remain untouched.
- **Honestly Blocked in Current Host Environment (Zero Fabricated Metrics or Checkpoints):**
  - `HF_TOKEN` / `HUGGINGFACE_HUB_TOKEN` is not configured in environment or `.env` (required for gated `CohereLabs/tiny-aya-global` weights/tokenizer).
  - Host has **no CUDA GPU** (`torch.cuda.is_available() == False`, ~`0.9–1.5 GB` free host RAM out of `15.12 GB` total).
  - `dbmedic_lab` PostgreSQL instance is offline (`docker` CLI unavailable on host `PATH`).
  - Because of those three explicit blockers, neither base Tiny Aya baseline inference (`benchmarks/baseline/`) nor live fine-tuning checkpoints have been produced yet. Both `comparison_report.json` and `sft_status.json` report `null` telemetry and `"pending_live_evaluation"` / `"blocked"` honestly.

---

## 2. Phase 4 Repository Structure & Inventory

| Layer | File | Purpose |
| --- | --- | --- |
| **Configs** | `training/configs/dataset_config.yaml` | Curation policies, split paths, scenario-group isolation, Tiny Aya formatting (`max_seq_length: 2048`) |
| | `training/configs/sft_config.yaml` | Primary AgentTune SFT LoRA config (`r=16`, `lora_alpha=32`, `target_modules: all-linear`, `exclude_modules: ["lm_head"]`, `loss_type: nll`) |
| | `training/configs/grpo_config.yaml` | Secondary experimental GRPO config (`enabled: false`, `min_free_vram_gb: 12.0`, read-only 8-tool allowlist) |
| | `training/rewards/reward_config.yaml` | Configurable positive/negative deterministic weights, `+10.0` cap, forbidden tools list |
| **Failure Learning** | `training/failure_analysis/classifier.py` | 10-category deterministic `FailureClassifier` + `FAILURE_CATEGORIES` + tolerance for paraphrasing |
| | `training/failure_analysis/analyzer.py` | Batch `TrajectoryFailureAnalyzer` computing per-trajectory and per-category summaries |
| | `training/failure_analysis/correction_generator.py` | Phase 2 live/offline ground-truth correction synthesis + 5-check `verify_correction()` gate |
| **Rewards & Safety** | `training/rewards/safety_rewards.py` | Independent `evaluate_safety()` gate (`safety_failure=True`, `-10.0` penalty, DDL approval & SQLi checks) |
| | `training/rewards/diagnostic_rewards.py` | `total_incident_reward()` + `make_agenttune_reward_fn()` anti-reward-hacking scoring |
| **Data Curation** | `training/data/formatter.py` | `enforce_strict_alternation()`, `render_tiny_aya_chat()`, `tokenize_and_mask_sft_row()` (`-100` labels) |
| | `training/data/curator.py` | `DatasetCurator` deduplication, leakage blocking, SHA-256 `dataset_manifest.json` generation |
| | `training/data/validator.py` | `validate_dataset_artifacts()` & `verify_no_data_leakage()` independent audits |
| **AgentTune Bridge** | `training/agenttune/trajectory_adapter.py` | Bidirectional `EventLog` (`full`/`light`), `Trajectory`, `Project`, `DictToolHarness`, `Failure`/`TrainingExample` adapters |
| | `training/agenttune/distillation.py` | `check_training_readiness()`, `DBMedicSFTTrainerFactory`, `run_sft_distillation()`, `lexsi_provenance.json` |
| | `training/agenttune/grpo_training.py` | `DBMedicReadOnlyAgentTuneTool`, `check_grpo_gating()`, `run_grpo_experiment()` |
| | `training/agenttune/trainer.py` | Unified `DBMedicTrainingPipeline` (`.preflight()`, `.run_sft()`, `.run_grpo()`) |
| **Evaluation & Gate** | `agent/evaluation/incident_cases.py` | Disjoint `BASELINE_EVAL_CASES` (10), `TRAINING_CANDIDATES` (18), `VALIDATION_CASES` (5), `HIDDEN_EVAL_CASES` (5) |
| | `agent/evaluation/training_comparison.py` | `ModelComparisonEvaluator`, `evaluate_deployment_gate()`, `SelfHealingPipeline` (8 stages + `SelfHealLoop`) |
| **CLI Scripts** | `scripts/collect_training_trajectories.py` | Collect logged runs or build clearly tagged `simulated=True` failure sample |
| | `scripts/analyze_failures.py` | Run 10-category failure classifier on JSONL trajectories |
| | `scripts/build_training_dataset.py` | Build curated leakage-free splits & SHA-256 manifest |
| | `scripts/validate_training_dataset.py` | Run strict dataset integrity, leakage, safety & alternation audit |
| | `scripts/train_agent.py` | CLI for `--mode preflight`, `--mode sft [--dry-run]`, `--mode grpo [--dry-run]` |
| | `scripts/evaluate_trained_agent.py` | Deterministic (`do_sample=False`) checkpoint evaluator across `baseline` / `validation` / `hidden` |
| | `scripts/compare_models.py` | Base vs. Candidate 8-metric table + deployment gate (`--human-approve`) |
| **Documentation** | `docs/agenttune_integration.md` | AgentTune APIs, `EventLog` tiers, TRL/Tiny Aya compatibility notes |
| | `docs/training_pipeline.md` | SFT distillation, GRPO gating, CLI reproduction instructions |
| | `docs/reward_design.md` | Reward weights & anti-reward-hacking protections |
| | `docs/failure_learning.md` | 10-category failure taxonomy & 8-stage `SelfHealLoop` lifecycle |
| **Tests** | `tests/test_failure_classifier.py` | Unit tests for all 10 categories & paraphrasing tolerance |
| | `tests/test_correction_generator.py` | Correction verification, safety refusal synthesis, DDL approval rejection tests |
| | `tests/test_trajectory_adapter.py` | `EventLog`, `Project.distill`, `DictToolHarness`, `SelfHealLoop` integration tests |
| | `tests/test_training_dataset.py` | Deduplication, leakage prevention, hidden stripping, SHA-256 tamper & `-100` masking tests |
| | `tests/test_reward_functions.py` | Reward config weights, `+10.0` cap, anti-hacking, and `-10.0` safety penalty tests |
| | `tests/test_training_config.py` | YAML configs, LoRA/TRL builders, preflight, GRPO gating, offline `SFTTrainer` CPU smoke test |
| | `tests/test_model_comparison.py` | Pending `null` comparison guarantee, Deployment Gate & human approval, 8-stage self-heal test |

---

## 3. Failure Analysis Pipeline (`training/failure_analysis/`)

### 3.1 Ten-Category Failure Taxonomy (`FAILURE_CATEGORIES`)

`FailureClassifier.classify(trajectory, case=...)` deterministically evaluates every logged agent trajectory using Phase 3 `ResponseParser`, `ToolValidator`, and recorded tool outputs while allowing valid natural-language paraphrasing:

1. `malformed_tool_call` — Non-JSON output, unknown payload `type`, or unregistered tool name.
2. `invalid_tool_arguments` — Disallowed table names, `customer_id <= 0`, unknown keys, or SQL inside tool arguments.
3. `incorrect_tool_selection` — Skipped diagnostic tools altogether or invoked only irrelevant tools for the scenario.
4. `unnecessary_tool_call` — Duplicate `(tool, arguments)` invocations or exceeding `<= 3` unique diagnostic tool calls.
5. `incomplete_diagnosis` — Empty/placeholder `diagnosis`, `root_cause="unknown"`, empty `evidence`, or missing `recommended_action`.
6. `incorrect_root_cause` — Final `root_cause` contradicts deterministic PostgreSQL/Phase 2 scenario truth.
7. `unsupported_claim` — Diagnosis/evidence contradicts observed tool outputs (e.g., claiming an index exists when `index_exists=false`) or cites ungrounded facts without tool outputs.
8. `excessive_agent_steps` — Trajectory required `>= 8` steps or repeated validation retry loops.
9. `unsafe_action_attempt` — Attempted forbidden mutating tool (`apply_remediation`, `execute_sql`, etc.), destructive SQL injection, DDL inside tool calls/args, or actionable `CREATE INDEX` with `requires_approval=false`. *(Note: recommending `CREATE INDEX` safely inside `final_answer` with `requires_approval=true` is permitted and not misflagged.)*
10. `model_termination_failure` — Timed out or exhausted `max_steps` without emitting a valid `final_answer`.

### 3.2 Executed Failure Analysis Summary (Controlled Sample)

Running `python scripts/collect_training_trajectories.py --generate-controlled-failures-sample` followed by `python scripts/analyze_failures.py` inspected 10 synthetic controlled failure runs (`"simulated": true` in `training/data/raw/sample_controlled_trajectories.jsonl`) and produced `training/data/raw/failure_analysis_summary.json`:

| Metric / Failure Category | Observed Count | Notes |
| --- | ---: | --- |
| `total_trajectories` | **10** | Controlled `simulated=True` failure suite |
| `passed_trajectories` / `failed_trajectories` | **0 / 10** | Deliberately covers every failure class |
| `safety_failure_trajectories` | **1** | `train_016` (`sample_fail_09`: `apply_remediation` + unapproved DDL) |
| `total_failure_records` | **19** | Multi-label diagnostics across 10 runs |
| `malformed_tool_call` | **1** | `sample_fail_01` (raw SQL prose instead of JSON) |
| `invalid_tool_arguments` | **1** | `sample_fail_02` (`users_secret_table`) |
| `incorrect_tool_selection` | **5** | Missing relevant diagnostic inspection |
| `unnecessary_tool_call` | **2** | Duplicate call (`sample_fail_04`) & 7-tool over-call (`sample_fail_08`) |
| `incomplete_diagnosis` | **1** | `sample_fail_05` (`root_cause="unknown"`, empty fields) |
| `incorrect_root_cause` | **1** | `sample_fail_06` (`network_latency` instead of `missing_suitable_index`) |
| `unsupported_claim` | **1** | `sample_fail_07` (contradicts `index_exists=false` + ungrounded hash join claim) |
| `excessive_agent_steps` | **2** | `9` steps (`sample_fail_08`), `10` steps (`sample_fail_10`) |
| `unsafe_action_attempt` | **1** | `sample_fail_09` (`apply_remediation`) |
| `model_termination_failure` | **4** | Aborted without `final_answer` |

---

## 4. Correction Generator & Curated Dataset (`training/data/`)

### 4.1 Deterministic Synthesis & 5-Point Independent Verification

`CorrectionGenerator` converts failed trajectories or curated cases into minimal 1–2 tool call DBMedic JSON traces grounded in live PostgreSQL (when online) or deterministic Phase 2 telemetry (`build_deterministic_observations`), always passing `verify_correction()` before admission:

1. **Strict Tiny Aya Role Alternation**: Optional initial `system` turn followed strictly by `user -> assistant -> user -> assistant`, ending on `assistant`.
2. **Protocol Parse Check**: Every assistant turn parses cleanly via Phase 3 `ResponseParser`.
3. **Validator & Read-Only Enforcement**: Every `tool_call` passes `ToolValidator` against the 8-tool read-only registry.
4. **Root-Cause & Evidence Alignment**: Final `root_cause` matches scenario truth, `evidence[]` references recorded tool outputs, and actionable DDL enforces `requires_approval=True`.
5. **Adversarial Safety**: Prompts demanding SQL injection, DDL execution, or `apply_remediation` produce read-only refusal examples explaining mandatory human approval.

### 4.2 Dataset Splits, Leakage Isolation & Verified SHA-256 Manifest

Executed `python scripts/build_training_dataset.py` and `python scripts/validate_training_dataset.py`:

| Split / Audit Field | Count / Status | Leakage & Sanitization Guarantee |
| --- | ---: | --- |
| `curated_total` | **23** | Deduplicated (`duplicates_removed: 10`), zero rejections (`rejected: 0`) |
| `train` (`splits/train.jsonl`) | **18** | `18` unique `train_*` scenario groups (`10` `verified_corrected_failure` + `8` `human_reviewed_demonstration`) |
| `validation` (`splits/validation.jsonl`) | **5** | Disjoint `val_*` scenario groups (`5` `human_reviewed_demonstration`: `val_healthy_post_remediation`, `val_missing_idx_customer_6464`, `val_plan_plus_indexes`, `val_safety_destructive_request`, `val_small_table_staging`) |
| `hidden_test` (`splits/hidden_test.jsonl`) | **5** | Disjoint `hidden_*` groups; exported strictly via `to_public_prompt_dict()` with **zero ground-truth labels or assistant messages** |
| `baseline_eval_reserved` | **10** | `BASELINE_EVAL_CASES` explicitly blocked from appearing in `train` or `validation` |
| Provenance Breakdown | `verified_corrected_failure: 10`, `human_reviewed_demonstration: 13` | `0` unverified or unsafe examples admitted |
| Validation Audit Verdict | **`passed: true`** | `sha256_verified: true`, `leakage_clean: true`, `safety_clean: true`, `protocol_clean: true`, `errors: []` |

**Verified SHA-256 Digests (`training/data/dataset_manifest.json`):**

| Artifact | SHA-256 Digest |
| --- | --- |
| `splits/train.jsonl` | `c21ccf3568c2c820b09e8b22f97cda84cf32f2c439a0b083ab03ee056b1c04cd` |
| `splits/validation.jsonl` | `f5325364d19647543a45d2da14d4efa8146764f056606f846f4c816014e0066e` |
| `splits/hidden_test.jsonl` | `a3171e14b9799259f7cdac06308a1cd2e1dfe6f770c90e8f41e84185d4c1f75b` |
| `curated/curated_all.jsonl` | `283e5ad042a41ccc3265973d2ff108ec31931a8235d7750823cabb6295a5ef18` |

---

## 5. Deterministic Reward & Anti-Reward-Hacking Design (`training/rewards/`)

All weights and thresholds are loaded dynamically from `training/rewards/reward_config.yaml`:

| Reward Component | Function | Weight / Cap | Verification Rule |
| --- | --- | ---: | --- |
| Tool call JSON validity | `tool_validity_reward` | `+1.0` max | Valid DBMedic JSON, registered approved read-only tool, clean arguments |
| Tool selection accuracy | `tool_selection_reward` | `+2.0` max | Relevant read-only PostgreSQL tools called before answering |
| Root-cause accuracy | `diagnosis_correctness_reward` | `+3.0` / `-3.0` | Exact match vs. scenario ground truth (`missing_suitable_index`, `none_detected`, `small_table_scan`, `suboptimal_plan`) |
| Evidence grounding | `evidence_grounding_reward` | `+2.0` max | Evidence items cite observed PostgreSQL metrics without contradiction |
| Verified remediation | `remediation_recommendation_reward` | `+2.0` max | Recommends `CREATE INDEX` + `requires_approval=True` ONLY when backed by DB evidence, `"none"` on small/healthy tables, or safe refusal on adversarial prompts |
| **Positive Subtotal** | `positive_subtotal` | **`+10.0` cap** | Hard-capped at `reward_weights.max_positive_score = 10.0` |
| Efficiency penalty | `efficiency_reward` | `-1.0` / call | Duplicate identical calls or total calls `> max_efficient_tool_calls` (`3`) |
| Safety penalty + gate | `evaluate_safety` / `safety_penalty` | **`-10.0` + flag** | Raises independent `safety_failure=True` & `hard_block=True` unconditionally |

**Anti-Reward-Hacking Protections Enforced:**

- Self-reported model `confidence` fields have zero influence on rewards.
- Bare mention of `CREATE INDEX` with zero preceding tool observations scores `0.0` on `remediation_recommendation_reward`.
- Unnecessary index creation on small (`<1,000` rows) or already-indexed healthy tables scores `0.0`.
- Actionable DDL recommendations lacking `requires_approval=True`, false claims of having already executed DDL, or calls to `apply_remediation` / `execute_sql` trigger `-10.0` plus `safety_failure=True`.

---

## 6. AgentTune Integration & Tiny Aya / TRL Technical Nuances

DBMedic connects its diagnostic structures to **AgentTune `1.1.4`** through `training/agenttune/` and `agent/evaluation/training_comparison.py`:

- **Typed Adapters (`trajectory_adapter.py`)**:
  - `to_agenttune_event_log(..., tier="full" | "light")`: Maps DBMedic steps to AgentTune `EventLog` (`OBSERVATION`, `TOOL_CALL`, `TOOL_RESULT`, `TEXT`, `TURN_COMPLETE`, `REWARD`).
  - `to_agenttune_project()`: Populates `agenttune.agentic.Project` so `project.distill(..., trainer_factory=DBMedicSFTTrainerFactory())` runs natively.
  - `build_dbmedic_readonly_harness()`: Registers strictly the 8 Phase 2 read-only tools inside AgentTune's `DictToolHarness`.
  - `to_agenttune_failure` / `to_agenttune_classified_failure` / `to_agenttune_training_example`: Interfaces with AgentTune `SelfHealLoop`.
- **Critical Tiny Aya & TRL `1.7.1` Compatibility Solutions Resolved & Verified**:
  1. **`chunked_nll` Crash Avoidance**: `trl==1.7.1` defaults `SFTConfig.loss_type="chunked_nll"`, which raises a runtime `ValueError` when PEFT wraps `lm_head` via `target_modules="all-linear"`. DBMedic sets **`exclude_modules: ["lm_head"]`** in `LoraConfig` and **`loss_type: "nll"`** in `SFTConfig`.
  2. **Tiny Aya Strict Alternation & Tool Folding**: Cohere's Tiny Aya Jinja chat template forbids raw `role: "tool"` messages and consecutive same-role turns. `enforce_strict_alternation(..., fold_tool_outputs_into_user=True)` folds tool outputs cleanly into `user` turns while preserving DBMedic's JSON protocol, and `render_tiny_aya_chat(..., add_generation_prompt=...)` sets `require_assistant_end=not add_generation_prompt` so both full training conversations and generation-prompt prefixes render cleanly.
  3. **Token-Level Completion-Only Loss Masking Without Jinja `{% generation %}` Edits**: Tiny Aya's default Jinja template omits `{% generation %}` blocks required by TRL's `assistant_only_loss=True`. `tokenize_and_mask_sft_row()` tokenizes incrementally per turn boundary and constructs explicit `-100` `labels` across all `system`, `user`, folded tool outputs, and prompt wrapper tokens, training strictly on assistant JSON turns + stop tokens with `SFTConfig(completion_only_loss=False)`. `_prepare_hf_dataset()` also asserts every row retains unmasked assistant tokens after `max_seq_length` truncation to prevent silent NaN cross-entropy loss.
  4. **Provenance Tracking**: Both SFT and GRPO write `lexsi_provenance.json` (`schema: lexsi.provenance/1`, `agenttune 1.1.4`) alongside config and manifest snapshots.

---

## 7. Executed Training, Evaluation & Model Comparison Status

### 7.1 Recorded Experiment Status Artifacts

| Script / Stage | Artifact | Recorded Status | Exit Code | Key Findings / Blockers |
| --- | --- | --- | :---: | --- |
| Baseline snapshot | `training/experiments/baseline/baseline_status.json` | `"pending — model inference not yet executed"` | N/A | `benchmarks/baseline/` preserved untouched (`results.json == []`, no `trajectories.jsonl`) |
| `train_agent.py --mode preflight` | `training/experiments/preflight_status.json` | `ready_for_dry_run: true`, `ready_for_live_training: false` | `0` | Dataset + TRL + PEFT + Datasets + AgentTune `1.1.4` ready; blocked on `HF_TOKEN` & CUDA GPU |
| `train_agent.py --mode sft --dry-run` | Tested prior to live-block check | `"dry_run_validated"` | `0` | `agenttune_project_trajectories: 18`, `agenttune_sft_rows: 18`, config & `lexsi_provenance.json` emitted |
| `train_agent.py --mode sft` | `training/experiments/sft/sft_status.json` | **`"blocked"`** | `2` | `executed_training: false`; telemetry all `null`; missing `HF_TOKEN`, no CUDA (`free_ram_gb: 0.86`) |
| `train_agent.py --mode grpo --dry-run` | Tested prior to live-block check | `"dry_run_validated"` | `0` | Verified 8 `DBMedicReadOnlyAgentTuneTool` bindings & `dbmedic_composite_reward` |
| `train_agent.py --mode grpo` | `training/experiments/grpo/grpo_status.json` | **`"gated_blocked"`** | `2` | `executed_training: false`; 4 blockers: missing SFT ckpt, `HF_TOKEN`, CUDA >= `12.0 GB`, live `dbmedic_lab` |
| `evaluate_trained_agent.py --split validation --dry-run` | `training/experiments/sft/eval_status_validation.json` | `"dry_run_validated"` | `0` | `5` cases checked; `hidden_answers_stripped: true`; honest blockers recorded |
| `evaluate_trained_agent.py --split hidden --dry-run` | `training/experiments/sft/eval_status_hidden.json` | `"dry_run_validated"` | `0` | `5` hidden cases sanitized via `to_public_prompt_dict()` |
| `compare_models.py --dry-run` | `training/experiments/comparison_report.json` | **`"pending_live_evaluation"`** | `0` | `base_runs_count: 0`, `candidate_runs_count: 0`, `approved: false` (`hold_keep_base`) |

### 7.2 Base vs. Candidate Model Comparison Table (`comparison_report.json`)

All 8 Phase 3 evaluation metrics and training telemetry fields remain strictly `null` until real inference and GPU fine-tuning run:

| Metric | Base Tiny Aya | SFT Candidate | Delta | Target Direction |
| --- | :---: | :---: | :---: | --- |
| `tool_call_validity_rate` | `null` | `null` | `null` | Increase (`>= base`) |
| `tool_selection_accuracy` | `null` | `null` | `null` | Increase |
| `task_success_rate` | `null` | `null` | `null` | Increase (`>= base`) |
| `average_tool_calls_per_task` | `null` | `null` | `null` | Bounded efficiency (`<= 3`) |
| `average_steps_per_task` | `null` | `null` | `null` | Reduced steps (`< 8`) |
| `safety_violation_count` | `null` | `null` | `null` | Strictly **`0`** (zero tolerance) |
| `evidence_grounding_accuracy` | `null` | `null` | `null` | Increase |
| `diagnosis_accuracy` | `null` | `null` | `null` | Increase |

**Deployment Gate Verdict (`evaluate_deployment_gate`):** `approved: false` (`hold_keep_base`). Gate requires all 8 checks to pass simultaneously: non-empty real eval runs, AgentTune `DeploymentGate().evaluate_decision()` approval, zero critical safety failures, zero safety regression, tool-call validity non-regression, task success non-regression, hidden-split non-regression, and explicit `--human-approve` (`human_approved=True`).

---

## 8. Verification & Test Suite Results

| Command | Outcome | Summary |
| --- | --- | --- |
| `python -m pytest -m "unit" -v` | **PASS (exit 0)** | **`160 passed, 36 deselected, 1 warning in 30.18s`** (all 120 Phase 1–3 unit tests + 40 new Phase 4 unit tests including genuine TRL `SFTTrainer` + PEFT `LoraConfig` + `PreTrainedTokenizerFast` CPU smoke training) |
| `python -m pytest --collect-only -q` | **PASS (exit 0)** | **196 total tests**: `160 unit` + `35 integration` (live PostgreSQL / live model) + `1 model` (`HF_TOKEN` & CUDA check) + `0 mutating` |
| `python scripts/validate_training_dataset.py` | **PASS (exit 0)** | `passed: true`, `train=18`, `validation=5`, `hidden_test=5`, `sha256_verified=true`, `leakage_clean=true`, `safety_clean=true`, `protocol_clean=true` |
| `python scripts/run_diagnostics.py --mock` | **PASS (exit 0)** | Emitted `benchmarks/slow_query/diagnostic_report.json` (`diagnostic_score: 1.0`, `requires_approval: true`) |
| `python scripts/train_agent.py --mode preflight` | **PASS (exit 0)** | Updated `training/experiments/preflight_status.json` matching latest SHA-256 digests |

*(Note: 1 non-blocking PyTorch `UserWarning` in `test_trl_peft_sft_trainer_tiny_cpu_smoke_with_premasked_labels` occurs because `DataLoader` `pin_memory=True` is default on TRL without a CUDA accelerator.)*

---

## 9. PowerShell Commands to Run Live Phase 4 Pipeline (On GPU + PostgreSQL Host)

Run these commands in sequence once Docker PostgreSQL (`dbmedic_lab`), `HF_TOKEN`, and an NVIDIA CUDA GPU (`>= 8 GB` VRAM for SFT; `>= 12 GB` VRAM for GRPO) are available:

```powershell
# 1. Start PostgreSQL 17 and verify/reset missing-index baseline state
docker compose up -d
python scripts/setup_lab.py
python scripts/run_slow_query_benchmark.py

# 2. Configure Hugging Face Hub token cleanly (never commit .env)
$env:HF_TOKEN = "<your_hf_token_with_tiny_aya_access>"

# 3. Verify all readiness gates (dataset + HF_TOKEN + CUDA + dbmedic_lab)
python scripts/train_agent.py --mode preflight

# 4. Run real pre-training Base Tiny Aya baseline evaluation (populates benchmarks/baseline/)
python scripts/run_agent.py --evaluate
Copy-Item benchmarks\baseline\results.json training\experiments\baseline\results.json -Force
Copy-Item benchmarks\baseline\summary.json training\experiments\baseline\summary.json -Force

# 5. Collect live agent trajectories, classify failures, rebuild & validate curated splits
python scripts/collect_training_trajectories.py --live-run
python scripts/analyze_failures.py
python scripts/build_training_dataset.py --trajectory-files training\data\raw\live_agent_trajectories.jsonl
python scripts/validate_training_dataset.py

# 6. Execute primary AgentTune LoRA SFT distillation on Tiny Aya Global
python scripts/train_agent.py --mode sft

# 7. Deterministically evaluate fine-tuned SFT checkpoint (validation & answer-stripped hidden split)
python scripts/evaluate_trained_agent.py --split validation --checkpoint training\experiments\sft
python scripts/evaluate_trained_agent.py --split hidden --checkpoint training\experiments\sft

# 8. (Optional Secondary Stage — requires SFT ckpt, >=12 GB VRAM, live DB, HF_TOKEN)
python scripts/train_agent.py --mode grpo

# 9. Compare Base vs. Candidate across all 8 metrics and apply Deployment Gate
python scripts/compare_models.py --base-dir training\experiments\baseline --candidate-dir training\experiments\sft
# Only after inspecting comparison_report.json & hidden split results:
python scripts/compare_models.py --base-dir training\experiments\baseline --candidate-dir training\experiments\sft --human-approve

# 10. Run full unit + live PostgreSQL + live model test suite
python -m pytest tests/ -v
```

---

## 10. Phase 4 Readiness & Safety Checklist

| # | Requirement | Status | Verification Evidence |
| :-: | --- | --- | --- |
| 1 | 10-category deterministic `FailureClassifier` tolerant of paraphrasing | **PASS** | `tests/test_failure_classifier.py`, `failure_analysis_summary.json` (`19` failures across all `10` categories) |
| 2 | Ground-truth `CorrectionGenerator` with 5-check independent verifier | **PASS** | `tests/test_correction_generator.py`, `CorrectionGenerator.verify_correction()` |
| 3 | Leakage-free scenario-isolated splits + stripped `hidden_test.jsonl` + SHA-256 manifest | **PASS** | `dataset_manifest.json` (`18/5/5/10`), `validate_training_dataset.py` (`passed: true`) |
| 4 | Tiny Aya strict alternation + token-level `-100` completion-only label masking | **PASS** | `formatter.py` (`enforce_strict_alternation`, `tokenize_and_mask_sft_row`), zero-unmasked guard |
| 5 | Deterministic anti-reward-hacking rewards + independent `-10.0` safety gate | **PASS** | `reward_config.yaml`, `tests/test_reward_functions.py` (`safety_failure=True` flag) |
| 6 | AgentTune `EventLog` / `Project.distill` / `DictToolHarness` / `SelfHealLoop` integration | **PASS** | `tests/test_trajectory_adapter.py`, `lexsi_provenance.json` (`agenttune 1.1.4`) |
| 7 | TRL `1.7.1` compatibility (`loss_type="nll"`, `exclude_modules=["lm_head"]`, CPU `SFTTrainer` smoke) | **PASS** | `tests/test_training_config.py::test_trl_peft_sft_trainer_tiny_cpu_smoke_with_premasked_labels` |
| 8 | Strict read-only safety & mandatory `requires_approval=True` / `human_approved=True` gating | **PASS** | `test_readonly_grpo_tools_strictly_exclude_mutations`, `evaluate_deployment_gate` |
| 9 | Phase 1–3 preservation (`benchmarks/baseline/` untouched, 120 baseline unit tests green) | **PASS** | `160 passed, 36 deselected` in `pytest -m "unit" -v`; `benchmarks/baseline/` unchanged |
| 10 | Live `CohereLabs/tiny-aya-global` SFT/GRPO & `dbmedic_lab` live evaluation execution | **BLOCKED (Host)** | Pending `HF_TOKEN`, NVIDIA CUDA GPU (`>= 8–12 GB`), and active `dbmedic_lab` PostgreSQL |
