# DBMedic Hackathon Live Demonstration Script (5–7 Minutes)

> **Honesty Guideline for Presenters:** Clearly distinguish between **live PostgreSQL deterministic execution**, **live Tiny Aya Global inference** (only when `HF_TOKEN` + GPU/RAM prerequisites are active), and **offline simulation / deterministic safety guards** (e.g. `AuditKITStub` and 10/10 validator red-team suite).

## Preparation (Before Going on Stage)

```powershell
# 1. Start PostgreSQL 17 and verify readiness
docker compose up -d
python scripts/dbmedic.py doctor

# 2. Reset dbmedic_lab to reproducible missing-index baseline
python demo/scripts/reset_demo_database.py
```

*(Fallback if presenting on a machine without Docker/PostgreSQL: pass `--mode simulated` to `dbmedic.py` / `run_complete_demo.py`, and verbally clarify to judges that the run uses DBMedic's explicit `simulated` mode while pointing to the preserved Phase 1 real PostgreSQL benchmarks.)*

---

## Live 5–7 Minute Presentation Flow

### Minute 0:00 – 1:00 | Environment & Problem Introduction
- Explain the 4 pillars DBMedic demonstrates: **Intelligence, Safety, Measurable Improvement, and Self-Healing Evaluation**.
- Show `python scripts/dbmedic.py doctor` output (`dbmedic_lab` connected, 200,000 rows in `orders`, `idx_orders_customer_id_exists=False`).

### Minute 1:00 – 2:30 | Diagnose Slow Query (Read-Only)
- Introduce the scenario: a customer lookup `SELECT COUNT(*) FROM orders WHERE customer_id = 4242` scans all 200,000 rows sequentially.
- Run:
  ```powershell
  # Use --mode tiny-aya if HF_TOKEN + GPU are active; otherwise explicit deterministic mode:
  python scripts/dbmedic.py diagnose --incident slow_query_001 --mode deterministic
  ```
- Point out:
  - Diagnostic evidence (`Sequential Scan`, ~24 ms median across 7 runs, zero suitable index).
  - If running `--mode tiny-aya`, highlight the model's read-only tool selection (`get_query_execution_plan`, `inspect_customer_id_index`).
  - Incident pauses safely at **`AWAITING_APPROVAL`** — **zero database mutations have occurred**.

### Minute 2:30 – 3:45 | Human-in-the-Loop Safety Approval
- Run:
  ```powershell
  python scripts/dbmedic.py approve --incident slow_query_001
  ```
- Show the operator review screen: exact SQL (`CREATE INDEX idx_orders_customer_id ON orders(customer_id);`), SHA-256 hash, target DB identity, expiration.
- Type your operator name and `APPROVE`. Briefly highlight why the model cannot self-approve (`approval_manager.py` blocks model actors and forged booleans).

### Minute 3:45 – 5:00 | Execute & Verify Before/After Improvement
- Run:
  ```powershell
  python scripts/dbmedic.py verify --incident slow_query_001
  ```
- Highlight **actual measured proof** rather than blind success claims:
  - Plan switches from **Sequential Scan** -> **Index Only Scan / Index Scan**.
  - Median latency drops dramatically (e.g., ~24 ms -> ~1.2 ms, ~95% faster).
  - Buffer blocks touched drop substantially.
  - Saved in `benchmarks/final_demo/remediation_comparison.json`.

### Minute 5:00 – 6:30 | Audit Trail, Training & Evaluation Honesty
- Run:
  ```powershell
  python scripts/dbmedic.py evaluate
  python scripts/dbmedic.py report
  ```
- Walk judges through:
  - `reports/audit_trail.jsonl` (SHA-256 chained lifecycle from `INCIDENT_CREATED` to `INCIDENT_CLOSED`).
  - Phase 4 AgentTune SFT/GRPO pipeline + curated 18/5/5 SHA-256 dataset (`validation.passed=True`).
  - Phase 5 AuditKIT evaluation (`10/10` deterministic red-team attacks blocked; honest `AuditKITStub` disclosure & `hold_keep_base` gate until live GPU fine-tuning runs).
- *(Optional 30s visual)* Open `streamlit run dashboard/app.py` to show the live dashboard panels.

