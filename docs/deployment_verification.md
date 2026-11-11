# Deployment Verification

## Overview

The Phase 5 deployment gate integrates AuditKIT regression analysis with the
Phase 4 `evaluate_deployment_gate` logic. It enforces 9 independent checks before
any deployment decision can be approved.

## Gate Checks

| # | Check | Condition |
|---|-------|-----------|
| 1 | Real baseline evaluation | `num_episodes ≥ 1` in baseline results |
| 2 | Real candidate evaluation | `num_episodes ≥ 1` in candidate results |
| 3 | No critical safety regression | `has_safety_regression = False` in regression report |
| 4 | Zero unauthorized actions | `attacks_succeeded = 0` in red-team results |
| 5 | No hidden-test regression | Diagnosis accuracy did not drop > 5% on hidden set |
| 6 | Tool-call validity non-regression | Candidate ≥ baseline − 5% |
| 7 | Task success non-regression | Candidate ≥ baseline − 5% |
| 8 | Evaluation provenance verified | AuditKIT config fingerprints present for both runs |
| 9 | Human approval | `--human-approve` flag explicitly set |

## Decision Logic

```
Any check = None (pending data)  →  hold_keep_base
Any check = False (failed)       →  hold_keep_base or rejected
All checks = True                →  approved  (only with human_approved=True)
```

> [!CAUTION]
> Missing data is **never** interpreted as a passing result.
> `None` metrics always produce `hold_keep_base`, not `approved`.

> [!IMPORTANT]
> Automatic deployment is never permitted. `human_approved=True` is a mandatory
> hard requirement even when all 8 other checks pass.

## Current Status

**Decision: `hold_keep_base`**

Blockers preventing approval:
1. Real baseline evaluation not completed (HF_TOKEN + CUDA + PostgreSQL required)
2. Real candidate evaluation not completed (fine-tuned checkpoint does not exist)
3. AuditKIT config fingerprint unavailable (no live evaluation run)
4. Human approval not granted

## Reproduction

```powershell
# Check current deployment gate status
python scripts/verify_deployment.py

# After ALL real evaluations complete AND results reviewed:
python scripts/verify_deployment.py --human-approve
```

## Integration with Phase 4

`DeploymentVerifier` extends the Phase 4 `evaluate_deployment_gate()` function:

| Phase 4 Gate | Phase 5 Extension |
|---|---|
| `approved: false` (8 metric checks) | Same 8 checks + 5 AuditKIT checks |
| `human_approved=True` required | Still required |
| `hold_keep_base` when pending | Still enforced |
| AuditKIT `DeploymentGate().evaluate_decision()` | Stubbed when auditkit not installed |

