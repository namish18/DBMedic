# Agent Architecture — DBMedic Phase 3

## Overview

DBMedic Phase 3 integrates **CohereLabs/tiny-aya-global** with the Phase 2 deterministic diagnostic tools to build a fully orchestrated autonomous diagnostic agent.

## Architecture Diagram

```
User Query
    │
    ▼
┌───────────────────────────────────────────────────────────┐
│                     DBMedicAgent                          │
│  (agent/core/agent.py — public interface)                 │
└───────────────────────────────┬───────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────┐
│                   AgentOrchestrator                       │
│  (agent/core/orchestrator.py — ReAct loop)                │
│                                                           │
│  1. Build messages (system prompt + user query)           │
│  2. Call model client → raw text                          │
│  3. Parse text → tool_call or final_answer                │
│  4. Validate tool call                                    │
│  5. Execute tool → observation                            │
│  6. Append observation to history                         │
│  7. Repeat until final answer or max steps                │
└──────┬────────────┬──────────────┬────────────┬───────────┘
       │            │              │            │
       ▼            ▼              ▼            ▼
  TinyAya     ResponseParser  ToolValidator  ToolRegistry
  Client        (parser)       (validator)   (registry)
(model_client)
```

## Component Responsibilities

### `agent/core/model_client.py` — TinyAyaClient

| Responsibility | Implementation |
|---|---|
| Load model from HF Hub | `AutoModelForCausalLM.from_pretrained` |
| Format messages | Model's official chat template via `apply_chat_template` |
| Generate responses | `model.generate()` with configurable params |
| Decode tokens | `tokenizer.decode(new_tokens_only)` |
| Handle credentials | HF_TOKEN from env only — never logged |
| Resource management | `unload()` clears GPU/CPU memory |

### `agent/core/tool_registry.py` — ToolRegistry

Registers 8 read-only Phase 2 diagnostic tools:

| Tool | Source Module | Purpose |
|---|---|---|
| `get_query_execution_plan` | `query_analyzer.py` | EXPLAIN plan retrieval |
| `analyze_query_performance` | `query_analyzer.py` | EXPLAIN ANALYZE with metrics |
| `get_table_statistics` | `table_monitor.py` | pg_stat_user_tables data |
| `get_table_size` | `table_monitor.py` | Storage size breakdown |
| `get_table_indexes` | `index_monitor.py` | Index definitions + usage |
| `get_index_usage` | `index_monitor.py` | pg_stat_user_indexes data |
| `inspect_customer_id_index` | `index_monitor.py` | orders.customer_id index check |
| `diagnose_slow_query` | `diagnosis_engine.py` | Full deterministic diagnosis |

> [!CAUTION]
> `apply_remediation` is **explicitly excluded** from the registry. The LLM cannot execute DDL.

### `agent/core/tool_validator.py` — ToolValidator

Validates every model-generated tool call before execution:

1. Tool name must be a non-empty string
2. Tool must exist in the registry
3. Arguments must be a JSON object
4. Session-level rate limit (default: 5 calls max)
5. Per-tool rate limit (default: 3 calls per tool)
6. Schema validation — type, enum, minimum
7. No unknown arguments
8. SQL injection detection in all argument values
9. Semantic validation — table names, template keys, customer IDs

### `agent/core/response_parser.py` — ResponseParser

Parses model-generated text into structured dicts:

- Handles clean JSON, JSON in markdown fences, embedded JSON blocks
- Validates `type` is `tool_call` or `final_answer`
- Validates required fields per response type
- Scans all string values for dangerous SQL patterns
- Returns `ParseError` on failure (never crashes)

### `agent/core/orchestrator.py` — AgentOrchestrator

Implements the full ReAct agent loop:

```
while step_index <= max_steps:
    response = model.generate(messages)
    parsed   = parser.parse(response.text)

    if parsed.type == "final_answer":
        return result

    elif parsed.type == "tool_call":
        validation = validator.validate(parsed)
        if not validation.is_valid:
            append rejection message to history
            continue

        output = registry.execute(tool, arguments)
        append tool output to history

    if tools_called >= max_tool_calls:
        force final answer request
```

## Safety Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   SAFETY BOUNDARIES                     │
├─────────────────────────────────────────────────────────┤
│ 1. Model output is UNTRUSTED INPUT                      │
│ 2. All tool calls go through ToolValidator              │
│ 3. Only registered tools may execute                    │
│ 4. No arbitrary SQL arguments accepted                  │
│ 5. apply_remediation NOT registered                     │
│ 6. Tool call limit per session (default: 5)             │
│ 7. Human approval mandatory for all DDL                 │
│ 8. Dangerous SQL patterns trigger rejection             │
└─────────────────────────────────────────────────────────┘
```

## Tool Call Protocol

The agent uses a strict JSON-only communication protocol.

### Tool Call
```json
{
  "type": "tool_call",
  "tool": "inspect_customer_id_index",
  "arguments": {},
  "reasoning": "Check if a suitable index exists for the customer_id filter"
}
```

### Final Answer
```json
{
  "type": "final_answer",
  "diagnosis": "Sequential scan detected on orders table due to missing index",
  "root_cause": "missing_suitable_index",
  "evidence": [
    "No idx_orders_customer_id index exists",
    "EXPLAIN shows Seq Scan on 200,000 rows"
  ],
  "recommended_action": "Create index on orders(customer_id)",
  "requires_approval": true,
  "confidence": 0.95
}
```

## Trajectory Recording

Every agent run records a full trajectory to `benchmarks/baseline/trajectories.jsonl`:

```json
{
  "incident_id": "slow_query_001_run1",
  "timestamp": "2026-11-10T14:00:00Z",
  "model_id": "CohereLabs/tiny-aya-global",
  "model_revision": "<sha>",
  "query": "Why is my orders query slow?",
  "success": true,
  "trajectory_steps": [...]
}
```

Trajectories are the input format for future AgentTune training.

> [!IMPORTANT]
> Credentials (HF_TOKEN, POSTGRES_PASSWORD, DATABASE_URL) are **never** written to trajectory files.

