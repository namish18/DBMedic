# DBMedic Safety, Human Approval & Audit Model

## 1. Defense-in-Depth Boundaries

DBMedic enforces strict separation between **autonomous diagnosis** (read-only) and **database modification** (human-authorized allowlist):

| Layer | Control Enforced |
|---|---|
| **Tool Registry** | Exposes only 8 read-only Phase 2 inspection tools to Tiny Aya; `apply_remediation` is never registered as a model tool. |
| **Response Parser & Tool Validator** | Rejects arbitrary SQL, unregistered tools, SQL injection patterns, per-session call overflow, and DDL keywords. |
| **Approval Manager** | Binds every approval to `(request_id, incident_id, action_id, sql_sha256, target_database, approved_by, confirmation_method, approval_digest)`. |
| **Remediation Manager** | Enforces an immutable allowlist (`create_idx_orders_customer_id` -> `CREATE INDEX idx_orders_customer_id ON orders(customer_id)`), blocks `DROP`/`TRUNCATE`/`ALTER`, checks DB identity immediately prior to DDL, and never auto-retries failures. |
| **Remediation Verifier** | Requires both catalog index verification AND execution-plan + benchmark validation before declaring `VERIFIED_IMPROVED`. |
| **Audit Logger** | Records append-only JSONL events in `reports/audit_trail.jsonl` with credential redaction and SHA-256 hash chaining. |

## 2. Human Approval Guarantees

- **No Self-Approval**: Actors matching AI/model/agent identities (`tiny_aya`, `model`, `agent`, `llm`, etc.) raise `ModelSelfApprovalError`.
- **No Raw Boolean Trust**: A raw model JSON field `"human_approved": true` without a persisted, SHA-256-bound human operator record is rejected.
- **SQL Tamper Detection**: Canonicalized DDL SHA-256 is computed at request creation and verified against the live DDL string before execution. Any mismatch invalidates the request (`SQLTamperingError`).
- **Expiration & Single Execution**: Expired approvals raise `ApprovalExpiredError`; executed approvals transition to `EXECUTED`, blocking duplicate runs (`DuplicateApprovalError`).
- **Target DB Revalidation**: Live `current_database()`, host, and port are checked immediately before executing DDL (`DatabaseIdentityMismatchError`).

## 3. Audit Trail Hash Chaining & Security Limitations

Each audit record in `reports/audit_trail.jsonl` includes `prev_event_hash` and `event_hash` (SHA-256 over canonical JSON).

**Documented Limitation (Honest Threat Model):**
Local file-based SHA-256 hash chaining detects accidental corruption and naive single-line edits, **but does not provide cryptographic tamper-proofing** against an adversary with write permissions to the host filesystem (who could regenerate all subsequent hashes). Production compliance deployments must ship audit events in real time to immutable remote WORM storage or sign events using an external KMS key.

Credentials (`HF_TOKEN`, `POSTGRES_PASSWORD`, DSN passwords) are scrubbed automatically by `redact_secrets()` before writing to logs, reports, or incident JSON files.

