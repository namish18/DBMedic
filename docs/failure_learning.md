# Failure Learning, Curated Dataset & Self-Healing Lifecycle

Phase 4 implements an end-to-end closed loop where failed DBMedic diagnostic trajectories are deterministically analyzed, repaired using Phase 2 ground truth, audited by safety and schema gates, curated into leakage-free splits, and threaded through AgentTune's `SelfHealLoop` and `DeploymentGate`.

## 10-Category Deterministic Failure Taxonomy (`training/failure_analysis/classifier.py`)

Every trajectory is classified across 10 failure modes using `ResponseParser`, `ToolValidator`, and PostgreSQL ground-truth rules while remaining tolerant of valid natural-language paraphrasing:

1. `malformed_tool_call`: Non-JSON output, unknown payload type, unregistered/hallucinated tool name.
2. `invalid_tool_arguments`: Unallowlisted tables, invalid `customer_id`, forbidden SQL keys in arguments.
3. `incorrect_tool_selection`: Only irrelevant/unrelated tools called relative to the incident question.
4. `unnecessary_tool_call`: Duplicate identical `(tool, arguments)` calls or excessive over-calling (`>4` valid tools).
5. `incomplete_diagnosis`: Terminated with empty/unknown `root_cause` or missing diagnosis/evidence fields.
6. `incorrect_root_cause`: Final `root_cause` disagrees with deterministic PostgreSQL scenario truth.
7. `unsupported_claim`: Evidence contradicts recorded tool outputs or makes diagnostic claims with zero tool evidence.
8. `excessive_agent_steps`: Exceeded allowed step budget (`num_steps > 7` or unproductive retry loops).
9. `unsafe_action_attempt`: Attempted mutating tool (`apply_remediation`), destructive SQL injection, DDL claim, or `requires_approval=False` on DDL.
10. `model_termination_failure`: Reached max steps/timeout without producing a valid `final_answer`.

## Ground-Truth Failure Correction (`training/failure_analysis/correction_generator.py`)

When a failure is detected, `CorrectionGenerator`:
- Queries live PostgreSQL Phase 2 diagnostic tools (when online) or reproducible deterministic incident fixtures (`offline=True` fallback labeled explicitly as `deterministic_phase2_fixture`).
- Constructs a complete, minimal 1–2 tool call multi-turn trajectory ending in a schema-compliant `final_answer` using Tiny Aya's strict `system -> user -> assistant -> user -> assistant` turn structure.
- For safety-adversarial prompts (SQL injection, forced `CREATE INDEX`, `apply_remediation` coercion), teaches the model to refuse mutation cleanly, perform read-only diagnostics, set `requires_approval: true`, and explain why human approval is mandatory.
- Runs `verify_correction()` verifying JSON parseability, `ToolValidator` acceptance, safety clearance, and root-cause accuracy before accepting any repair.

## Curated Dataset & Zero-Leakage Splits (`training/data/curator.py` & `validator.py`)

All curated data resides under `training/data/` with SHA-256 verification in `dataset_manifest.json`:
- **Scenario-Group Isolation**: Disjoint `scenario_group` identifiers guarantee zero overlap across `train.jsonl`, `validation.jsonl`, `hidden_test.jsonl`, and reserved `BASELINE_EVAL_CASES`.
- **Hidden Set Sanitization**: `hidden_test.jsonl` exports `IncidentCase.to_public_prompt_dict()` ONLY — stripping `expected_root_cause`, `expected_tools_to_call`, `expected_action`, and assistant messages so ground truth can never leak into training or prompts.
- **Immutable Baseline Snapshot**: `benchmarks/baseline/` is preserved untouched and excluded from `train` and `validation` splits.

## 8-Stage Self-Healing Lifecycle (`agent/evaluation/training_comparison.py`)

`SelfHealingPipeline` connects:
1. **COLLECT**: Gather logged or sample trajectories via `TrajectoryLogger` / `collect_training_trajectories.py`.
2. **CLASSIFY**: Run 10-category `FailureClassifier` + AgentTune `Failure` conversion.
3. **GENERATE CORRECTION**: Synthesize deterministic DBMedic repairs + AgentTune `TrainingExample` preference pairs via `SelfHealLoop`.
4. **VERIFY CORRECTION**: Reject any unverified, malformed, or unsafe correction (`evaluate_safety`, `verify_correction`).
5. **ADD TO CURATED DATASET**: Deduplicate, enforce strict Tiny Aya alternation, compute `-100` masks, write SHA-256 verified splits.
6. **RETRAIN CANDIDATE**: Invoke `DBMedicTrainingPipeline` / `Project.distill()`.
7. **EVALUATE CANDIDATE**: Measure all 8 Phase 3 metrics deterministically (`do_sample=False`).
8. **DEPLOYMENT GATE**: Run `DeploymentGate().evaluate_decision(...)` + DBMedic safety/hidden-split non-regression checks + mandatory `human_approved=True`.

