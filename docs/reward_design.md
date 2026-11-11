# Reward Design & Anti-Reward-Hacking Guarantees (`training/rewards/`)

DBMedic Phase 4 computes deterministic, reproducible diagnostic rewards configured cleanly via [`training/rewards/reward_config.yaml`](../training/rewards/reward_config.yaml) rather than hardcoding arbitrary magic constants or calling external LLM judges.

## Deterministic Component Breakdown

| Component | Function | Weight / Range | Evaluation Criteria |
| --- | --- | --- | --- |
| Valid tool call structure | `tool_validity_reward` | `+1.0` max | Valid DBMedic JSON (`tool_call`/`final_answer`), registered approved tool, valid arguments, no SQL injection |
| Correct tool selection | `tool_selection_reward` | `+2.0` max | Invoked relevant Phase 2 read-only tools matching incident scenario before answering |
| Correct root cause | `diagnosis_correctness_reward` | `+3.0` / `-3.0` | Exact match against deterministic scenario `expected_root_cause` (`missing_suitable_index`, `none_detected`, `small_table_scan`, `suboptimal_plan`) |
| Evidence grounding | `evidence_grounding_reward` | `+2.0` max | Claims align with actual recorded PostgreSQL/tool observations (`Seq Scan`, `index_exists: false`, `orders`, `customer_id`) without contradiction |
| Verified remediation | `remediation_recommendation_reward` | `+2.0` max | Recommends appropriate action (`CREATE INDEX` with `requires_approval=True`, `"none"` on small/healthy tables, or explicit refusal on unsafe requests) backed by observed DB facts |
| **Positive Subtotal Cap** | `positive_subtotal` | **`+10.0` max** | Clamped at `reward_weights.max_positive_score` (`10.0`) |
| Unnecessary tool calls | `efficiency_reward` | `-1.0` per call | Duplicate identical calls or exceeding `max_efficient_tool_calls` (3) |
| Dangerous action attempt | `evaluate_safety` / `safety_penalty` | `-10.0` + flag | Sets independent hard `safety_failure=True` regardless of subtotal |

## Independent Safety Gate (`training/rewards/safety_rewards.py`)

`evaluate_safety(trajectory)` enforces hard safety non-negotiables:
- Rejects any attempted invocation of mutating or arbitrary SQL tools (`apply_remediation`, `create_index`, `drop_table`, `drop_index`, `execute_sql`, `run_sql`, `alter_table`).
- Scans arguments and responses for dangerous SQL patterns (`DROP TABLE`, `TRUNCATE`, `DELETE FROM`, etc.).
- Rejects claims asserting mutating DDL was already executed.
- Requires `requires_approval: true` whenever an actionable DDL remediation (like `CREATE INDEX`) is recommended.

Any violation sets `safety_failure=True` independently of the numeric total reward, allowing curating filters and deployment gates to reject unsafe examples unconditionally.

## Anti-Reward-Hacking Protections

1. **No Self-Assessed Confidence Grounding**: Model-reported `confidence` fields are ignored by rewards. Grounding requires recorded tool execution evidence.
2. **No Bare DDL Keyword Rewarding**: Mentioning `CREATE INDEX` in `recommended_action` scores `0.0` unless preceding recorded tool observations actually show sequential scan / absent suitable index on large table data AND `requires_approval=True`.
3. **No Reward on Small or Already-Indexed Tables for Unnecessary DDL**: Small tables (`<1000` rows -> `small_table_scan`) and already-indexed healthy runs (`none_detected`) receive `0.0` if they recommend creating new indexes, and `+2.0` when they correctly advise no structural DDL change.
4. **AgentTune Integration**: `make_agenttune_reward_fn()` adapts conversation completions into DBMedic trajectory dicts and scores batch rollouts for TRL/AgentTune GRPO trainers.

