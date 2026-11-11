# AgentTune Integration Guide (DBMedic Phase 4)

DBMedic Phase 4 integrates **AgentTune v1.1.4** with **CohereLabs/tiny-aya-global** while strictly preserving Phase 1–3 safety guarantees, deterministic PostgreSQL diagnostic tooling, and strict JSON communication protocols.

## Adapter Architecture (`training/agenttune/trajectory_adapter.py`)

AgentTune uses typed trajectory and lifecycle abstractions (`EventLog`, `Trajectory`, `Step`, `Project`, `DictToolHarness`, `Failure`, `ClassifiedFailure`, `TrainingExample`, `SelfHealLoop`, `DeploymentGate`). DBMedic bridges its own Phase 3 structures via:

- `to_agenttune_event_log(trajectory, tier="full" | "light")`: Maps DBMedic `AgentResult` / JSONL records into AgentTune `EventLog` (`OBSERVATION`, `TOOL_CALL`, `TOOL_RESULT`, `TEXT`, `TURN_COMPLETE`, `REWARD`).
  - `tier="full"` is required for `Project.distill()` / trainable teacher traces.
  - `tier="light"` is used for observational monitoring, failure detection, and audit inspection.
- `event_log_to_dbmedic_messages(log)`: Converts an `EventLog` back into Tiny Aya's strict alternating `system -> user -> assistant -> user -> assistant` message sequence with tool outputs folded into `user` turns and assistant tool invocations formatted in DBMedic JSON (`{"type": "tool_call", "tool": ..., "arguments": ...}`).
- `to_agenttune_project(trajectories, tier="full")`: Populates AgentTune's `Project` lifecycle container so `project.sft_dataset("sft")` and `project.distill(...)` work directly.
- `build_dbmedic_readonly_harness()`: Exposes ONLY the 8 approved Phase 2 read-only tools (`get_query_execution_plan`, `analyze_query_performance`, `get_table_statistics`, `get_table_size`, `get_table_indexes`, `get_index_usage`, `inspect_customer_id_index`, `diagnose_slow_query`) inside `DictToolHarness`. Mutating tools (`apply_remediation`, `execute_sql`, arbitrary DDL) are strictly excluded.

## Tiny Aya Chat Template & TRL Compatibility Constraints

1. **Strict Role Alternation**: `CohereLabs/tiny-aya-global`'s Jinja chat template requires optional initial `system` turn followed strictly by alternating `user` and `assistant` turns. Native `"tool"` roles raise Jinja template errors unless folded into `"user"` turns via `enforce_strict_alternation()`.
2. **Completion-Only Loss Masking**: Because Tiny Aya's default Jinja template does not contain `{% generation %}` tags required by TRL's `assistant_only_loss=True`, and TRL `1.7.1`'s `completion_only_loss=True` only targets prompt+completion columns, DBMedic pre-tokenizes conversations via `tokenize_and_mask_sft_row()`, setting `labels = -100` across all `system`, `user`, and folded tool-result tokens and preserving target IDs exclusively on assistant turns (including stop tokens).
3. **`loss_type="nll"` + `exclude_modules=["lm_head"]`**: `trl==1.7.1` defaults `SFTConfig.loss_type` to `"chunked_nll"`, which raises `ValueError` if `lm_head` is wrapped by LoRA (`target_modules="all-linear"`). Both `sft_config.yaml` and `DBMedicSFTTrainerFactory` enforce `loss_type="nll"` and `exclude_modules=["lm_head"]`.

## Provenance Tracking

Every SFT and GRPO experiment run writes:
- `lexsi_provenance.json` via `agenttune.utils.provenance.write_provenance(...)`
- `training_config.yaml` / `grpo_config_snapshot.yaml`
- `dataset_manifest_snapshot.json`
- Honest `sft_status.json` / `grpo_status.json` reporting execution state (`dry_run_validated`, `blocked`, `gated_blocked`, or `completed`) with zero fabricated metrics.

