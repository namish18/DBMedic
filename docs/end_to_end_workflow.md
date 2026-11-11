# DBMedic End-to-End Incident Workflow

This guide documents the complete operational lifecycle managed by `IncidentOrchestrator` (`agent/core/incident_orchestrator.py`) and driven via `scripts/dbmedic.py`.

## 1. Incident Lifecycle State Machine

```text
CREATED ──▶ INVESTIGATING ──▶ DIAGNOSED ──▶ AWAITING_APPROVAL ──▶ APPROVED ──▶ REMEDIATING ──▶ VERIFYING ──▶ RESOLVED
                 │               │                  │                │              │             │
                 ▼               ▼                  ▼                ▼              ▼             ▼
              BLOCKED    NEEDS_INVESTIGATION    REJECTED ──▶ RESOLVED          FAILED   NEEDS_INVESTIGATION
```

Every transition passes through `IncidentOrchestrator.transition_state()`, which raises `InvalidStateTransitionError` and writes a `STATE_TRANSITION_REJECTED` audit event on any illegal transition (such as skipping `AWAITING_APPROVAL` directly to `REMEDIATING`, or modifying a `RESOLVED` incident).

## 2. Step-by-Step CLI Workflow (Windows PowerShell)

### Step 1: Check Readiness
```powershell
python scripts/dbmedic.py doctor
```
Generates `reports/environment_readiness.json` with `READY`, `BLOCKED`, and `OPTIONAL` statuses for Python packages, `dbmedic_lab`, Tiny Aya, CUDA/RAM, AgentTune, and AuditKIT (`AuditKITStub` vs real package).

### Step 2: Diagnose Incident & Propose Remediation
```powershell
# Explicit deterministic diagnosis against live dbmedic_lab:
python scripts/dbmedic.py diagnose --incident slow_query_001 --mode deterministic

# Or live Tiny Aya mode (fails cleanly if HF_TOKEN/GPU/DB missing):
python scripts/dbmedic.py diagnose --incident slow_query_001 --mode tiny-aya
```
1. Creates incident in `CREATED` -> collects pre-remediation `EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)`, table stats, index status, and 7-run benchmark median.
2. Runs diagnosis -> validates evidence alignment -> proposes `CREATE INDEX idx_orders_customer_id ON orders(customer_id)`.
3. Creates a time-bounded approval request bound to target DB + SQL SHA-256 and stops safely in `AWAITING_APPROVAL`. **Zero database modifications happen during `diagnose`.**

### Step 3: Human Operator Review & Approval
```powershell
python scripts/dbmedic.py approve --incident slow_query_001
```
Displays root cause, baseline latency/scan type, target DB, action ID, SQL SHA-256, and proposed SQL, then requires an explicit human operator confirmation (`APPROVE` or `REJECT`). Rejected requests transition safely to `REJECTED` -> `RESOLVED` with zero DDL executed.

### Step 4: Execute Approved DDL & Verify Performance Gain
```powershell
python scripts/dbmedic.py verify --incident slow_query_001
```
1. Revalidates stored human approval, expiration, SQL integrity hash, and live PostgreSQL database identity (`current_database()` / port).
2. Calls Phase 1 `apply_remediation()` once (if `idx_orders_customer_id` already exists, safely records idempotent detection without re-running DDL).
3. Captures fresh 7-run benchmark + `EXPLAIN (ANALYZE, BUFFERS)` + index definition (`REMEDIATING -> VERIFYING`).
4. Evaluates **both** execution plan change and measured latency before declaring success (`benchmarks/final_demo/remediation_comparison.json`).

### Step 5: Unified Evaluation & Report
```powershell
python scripts/dbmedic.py evaluate
python scripts/dbmedic.py report
```
Connects Phase 3–6 results into `benchmarks/final_demo/final_metrics.json`, `reports/final_summary.json`, and `reports/final_summary.md`.

