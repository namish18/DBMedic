# DBMedic Final System Architecture (Phases 1–6)

## 1. End-to-End Architecture Overview

DBMedic connects six incremental layers into a single auditable PostgreSQL self-healing workflow:

```text
  PostgreSQL Lab (dbmedic_lab)               Operator CLI / Dashboard
  [orders: 200k rows, customer_id]           [scripts/dbmedic.py | dashboard/app.py]
            │                                            │
            ▼                                            ▼
  Phase 2 Read-Only Tools                   Phase 6 Incident Orchestrator
  - query_analyzer.py                       - CREATED -> INVESTIGATING -> DIAGNOSED
  - table_monitor.py                        - AWAITING_APPROVAL -> APPROVED
  - index_monitor.py                        - REMEDIATING -> VERIFYING -> RESOLVED
  - diagnosis_engine.py                     - Modes: live_tiny_aya | deterministic | simulated
            │                                            │
            ▼                                            ▼
  Phase 3 Tiny Aya Agent                    Phase 6 Human Approval Gate
  - DBMedicAgent / AgentOrchestrator        - approval_manager.py
  - ToolRegistry (8 read-only tools)        - SQL SHA-256 + DB identity binding
  - ToolValidator & ResponseParser          - Blocks model self-approval & booleans
            │                                            │
            ▼                                            ▼
  Phase 4 & 5 Training & Evaluation         Phase 1 / 6 Gated Remediation & Verifier
  - AgentTune SFT/GRPO & Failure Analysis   - remediation_manager.py -> apply_remediation()
  - AuditKITAdapter / AuditKITStub          - remediation_verifier.py (before/after plan+latency)
  - RegressionAnalyzer & DeploymentVerifier - audit_logger.py (SHA-256 chain)
```

## 2. Phase Breakdown

| Phase | Purpose | Key Modules |
|---|---|---|
| **Phase 1** | Controlled PostgreSQL 17 Lab & Incident Baseline | `database/schemas/schema.sql`, `database/scenarios/slow_query.py`, `scripts/setup_lab.py` |
| **Phase 2** | Deterministic Diagnostic Engine (read-only) | `agent/tools/{query_analyzer,table_monitor,index_monitor}.py`, `agent/core/diagnosis_engine.py` |
| **Phase 3** | Autonomous Agent (`CohereLabs/tiny-aya-global`), Validation & Eval | `agent/core/{agent,orchestrator,model_client,tool_registry,tool_validator,response_parser}.py`, `agent/evaluation/` |
| **Phase 4** | AgentTune SFT Distillation, 10-Category Failure Analysis & Rewards | `training/{failure_analysis,data,rewards,agenttune}/`, `agent/evaluation/training_comparison.py` |
| **Phase 5** | AuditKIT Adapter (`AuditKITStub` fallback), Red-Team & 9-Check Gate | `agent/audit/`, `agent/evaluation/{auditkit_evaluator,auditkit_scenarios,deployment_verifier}.py` |
| **Phase 6** | Unified Product CLI, Lifecycle State Machine, Approval/Audit & Demo | `agent/core/{environment_check,incident_orchestrator,approval_manager,remediation_manager,remediation_verifier,audit_logger}.py`, `scripts/dbmedic.py`, `demo/`, `dashboard/` |

## 3. Execution Modes & Zero Silent Fallback

1. **`live_tiny_aya`**: Invokes `DBMedicAgent` with `CohereLabs/tiny-aya-global` over `dbmedic_lab`. Strictly refuses to run or silently fall back if `HF_TOKEN`, hardware capacity, or PostgreSQL is missing — transitions incident cleanly to `BLOCKED` or `FAILED`.
2. **`deterministic`**: Invokes Phase 2 deterministic `diagnose_slow_query()` + `run_benchmark()` against live `dbmedic_lab` without requiring GPU weights.
3. **`simulated`**: Explicit offline synthetic execution (`simulated: true`) used strictly for offline demonstrations or CI/sandbox validation when PostgreSQL is unavailable.

## 4. Artifact Locations

- `reports/environment_readiness.json` — Hardware, package, PostgreSQL, Tiny Aya, AgentTune & AuditKIT status
- `reports/audit_trail.jsonl` — Append-only SHA-256 chained audit trail (secret-redacted)
- `reports/incidents/<incident_id>.json` — Persisted incident lifecycle records
- `reports/approvals.json` — Stored human approval authorizations with SQL SHA-256 digests
- `benchmarks/final_demo/remediation_comparison.json` — Before/after plan & latency metrics
- `benchmarks/final_demo/final_metrics.json` — Unified Phases 3–6 evaluation metrics
- `benchmarks/baseline/` — Immutable Phase 3 baseline artifacts (preserved untouched)

