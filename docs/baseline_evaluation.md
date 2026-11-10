# Baseline Evaluation — DBMedic Phase 3

## Overview

This document describes the **pre-training baseline evaluation** of the DBMedic agent using `CohereLabs/tiny-aya-global`.

The baseline establishes the model's starting diagnostic performance **before any AgentTune training**, providing the reference point for measuring training improvement.

> [!IMPORTANT]
> Do not run AgentTune training before capturing baseline results. The baseline is the control group.

## Evaluation Dataset

### Dataset Structure

| Split | Cases | Purpose |
|---|---|---|
| `baseline` | 10 | Measured baseline evaluation |
| `training` | 2 | AgentTune training candidates |
| `hidden` | 2 | Unbiased held-out evaluation |

### Baseline Cases

All 10 baseline cases target the same incident (**missing index on orders.customer_id**) but vary:

- **Question phrasing**: canonical, index-focused, size-focused, explicit SQL context
- **Customer ID**: 4242, 1234, 999, 7777 (different filter values, same schema)
- **Tool selection framing**: questions that bias toward different tool selections

### Ground Truth

Ground truth is **deterministic PostgreSQL observations**, not model outputs:

| Fact | Source | Value |
|---|---|---|
| Scan type | EXPLAIN output | Sequential Scan |
| Index exists | pg_indexes | False (pre-remediation) |
| Table row count | pg_class | ~200,000 |
| Expected root cause | Diagnosis engine | `missing_suitable_index` |

## Measured Metrics

### 1. Tool-Call Validity Rate
**Definition**: Fraction of model-generated tool calls that pass all validation checks.

```
valid_calls / (valid_calls + rejected_calls)
```

**Ideal**: 1.0 (all calls valid)

### 2. Tool Selection Accuracy
**Definition**: Fraction of runs where the agent called at least one relevant diagnostic tool.

Relevant tools: `get_query_execution_plan`, `analyze_query_performance`, `inspect_customer_id_index`, `get_table_indexes`, `diagnose_slow_query`

**Ideal**: 1.0

### 3. Task Success Rate
**Definition**: Fraction of runs producing a valid non-trivial final answer.

Requires: `success=True`, non-empty diagnosis, `root_cause != "unknown"`.

**Ideal**: 1.0

### 4. Average Tool Calls per Task
**Definition**: Mean number of tools called across all runs.

**Expected baseline**: 1–3 (simple task, one right tool call is sufficient)

### 5. Average Agent Steps
**Definition**: Mean number of orchestrator steps (model calls + tool executions + retries).

### 6. Safety Violation Count
**Definition**: Total dangerous or destructive tool calls rejected across all runs.

**Ideal**: 0 (no safety violations)

### 7. Evidence Grounding Accuracy
**Definition**: Fraction of evidence items in the final answer that reference observable facts from tool outputs (not invented).

**Ideal**: 1.0 (all evidence grounded in tool observations)

### 8. Final Diagnosis Accuracy
**Definition**: Fraction of runs correctly identifying `missing_suitable_index` as root cause.

**Ideal**: 1.0

## Running Baseline Evaluation

### Prerequisites

1. Docker container running (`docker compose up -d`)
2. HF_TOKEN set in `.env`
3. Model access accepted on HF Hub
4. `transformers`, `torch`, `accelerate` installed

### Quick test (3 cases, verifies pipeline works)

```powershell
.venv\Scripts\python.exe scripts/run_agent.py --evaluate --max-cases 3
```

### Full baseline (all 10 cases)

```powershell
.venv\Scripts\python.exe scripts/run_agent.py --evaluate
```

### Programmatic evaluation

```python
from agent.core.agent import DBMedicAgent
from agent.evaluation.evaluator import BaselineEvaluator
from agent.evaluation.incident_cases import BASELINE_EVAL_CASES

agent = DBMedicAgent()
agent.initialize()

evaluator = BaselineEvaluator(agent=agent)
summary = evaluator.run_baseline()
```

## Output Files

| File | Description |
|---|---|
| `benchmarks/baseline/results.json` | Per-case structured results |
| `benchmarks/baseline/trajectories.jsonl` | Full JSONL trajectory log |
| `benchmarks/baseline/summary.json` | Aggregate metrics summary |

## Reproducing Baseline Results

To reproduce the exact baseline:

1. Use the same model revision (SHA recorded in `summary.json`)
2. Use `do_sample=False` (deterministic generation)
3. Use the same generation config from `summary.json`
4. Ensure the database is in pre-remediation state (no `idx_orders_customer_id`)

```powershell
# Verify pre-remediation state
.venv\Scripts\python.exe -c "
import psycopg, os
conn = psycopg.connect(os.getenv('LAB_DATABASE_URL'))
cur = conn.cursor()
cur.execute(\"SELECT 1 FROM pg_indexes WHERE indexname = 'idx_orders_customer_id'\")
exists = cur.fetchone() is not None
print('Index exists:', exists, '(should be False for baseline)')
"
```

## Interpreting Results

The baseline results will reflect the model's **zero-shot** diagnostic capability.

Expected baseline behavior for a small instruction-tuned model:

- Task success rate: 20–60% (varies significantly by model size and prompt quality)
- Diagnosis accuracy: 10–40% (harder — requires correct root_cause string)
- Tool selection accuracy: 30–70% (model may select irrelevant or no tools)
- Safety violations: 0 (strict validator prevents dangerous calls)

After AgentTune training (Phase 4), these numbers should improve significantly.

## Limitations

- CPU inference is very slow (~30–120 s/response); a full 10-case evaluation may take 1–3 hours on CPU
- The model's small size limits multi-step reasoning ability at baseline
- JSON output quality varies; parse retries are counted as extra steps

