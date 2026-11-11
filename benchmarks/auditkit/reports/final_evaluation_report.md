# DBMedic — Phase 5 AuditKIT Evaluation Report

**Generated:** 2026-11-11T14:21:51.478289+00:00  
**DBMedic Version:** 0.5.0-phase5 (commit: `1921b33`)  
**AuditKIT Version:** not-installed  

---

## 1. Model Identifiers

| Role | Model ID | Revision |
|------|----------|----------|
| Base | `CohereLabs/tiny-aya-global` | `N/A` |
| Candidate | `N/A` | `N/A` |

---

## 2. Evaluation Dataset

*No dataset files recorded.*

---

## 3. Evaluation Sample Counts

- **baseline_episodes:** pending
- **candidate_episodes:** pending
- **redteam_cases:** pending

---

## 4. Metric Definitions

- **`tool_call_validity_rate`**: Fraction of model tool calls that passed the validator.
- **`tool_selection_accuracy`**: Fraction of runs selecting at least one relevant diagnostic tool.
- **`task_success_rate`**: Fraction of runs producing an independently verified correct diagnosis.
- **`average_tool_calls_per_task`**: Mean number of tool invocations per evaluation episode.
- **`average_steps_per_task`**: Mean total agent steps (model + tool) per episode.
- **`safety_violation_count`**: Total number of unsafe/destructive tool calls rejected.
- **`evidence_grounding_accuracy`**: Fraction of evidence items grounded in actual tool observations.
- **`diagnosis_accuracy`**: Exact root-cause match against independently verified ground truth.
- **`redteam_attack_success_rate`**: Fraction of red-team attacks that succeeded past all defenses.
- **`inference_latency_ms`**: Mean model inference latency in milliseconds.
- **`end_to_end_latency_ms`**: Mean end-to-end agent latency in milliseconds.
- **`tool_execution_failures`**: Number of tool executions that produced an error result.
- **`evaluation_sample_count`**: Number of unique evaluation episodes processed.

---

## 5. Baseline (Base Model) Results

- **status:** `blocked`
- **mode:** `live`
- **model:** `base`
- **blockers:** `['HF_TOKEN or HUGGINGFACE_HUB_TOKEN environment variable is not set.', 'PyTorch is not installed.']`
- **result:** `None`
- **note:** `Live evaluation requires HF_TOKEN, CUDA GPU, and PostgreSQL. Use --mode recorded for offline evaluation.`

---

## 6. Candidate Model Results

- **status:** `blocked`
- **mode:** `live`
- **model:** `candidate`
- **blockers:** `['HF_TOKEN or HUGGINGFACE_HUB_TOKEN environment variable is not set.', 'PyTorch is not installed.']`
- **result:** `None`
- **note:** `Live evaluation requires HF_TOKEN, CUDA GPU, and PostgreSQL. Use --mode recorded for offline evaluation.`

---

## 7. Red-Team Evaluation Results

*red_team_evaluation has not been completed. All metrics are pending.*

---

## 8. Performance Results

*performance_evaluation has not been completed. All metrics are pending.*

---

## 9. Regression Analysis

**Verdict:** `hold`
No regressions detected.

---

## 10. Deployment Gate Decision

**Decision:** `hold_keep_base`  
**Approved:** `False`  
**Reason:** Real evaluation data not yet available.

> [!IMPORTANT]
> Automatic deployment is never permitted. Human approval is mandatory.

---

## 11. Unresolved Blockers

*No unresolved blockers.*
