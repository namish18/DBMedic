# AuditKIT Integration Architecture

## Overview

Phase 5 integrates [Lexsi Labs AuditKIT](https://github.com/Lexsi-Labs/AuditKIT) as the
independent evaluation spine for DBMedic. AuditKIT provides a unified `Dataset → Adapter →
Model → Metrics → RunResult` pipeline with automatic SHA-256 configuration fingerprinting.

**License:** AuditKIT is released under LSAL-1.2 (source-available, non-commercial).
This adapter contains only DBMedic-specific integration code and does not redistribute
AuditKIT source.

## Architecture Diagram

```
DBMedic Trajectories / AgentTune EventLog
            │
            ▼
   ┌─────────────────────┐
   │   EpisodeBuilder    │  ← from_trajectory_file()
   │                     │  ← from_incident_cases()
   │                     │  ← from_agenttune_eventlog()
   └────────┬────────────┘
            │  Episode objects
            ▼
   ┌─────────────────────┐
   │  AuditKITAdapter    │  ← trajectory_to_episode()
   │                     │  ← evaluate_episodes(mode="recorded")
   │  ┌───────────────┐  │
   │  │ AuditKIT real │  │  (if pip install auditkit succeeds)
   │  │ AuditKITStub  │  │  (offline fallback — identical interface)
   │  └───────────────┘  │
   └────────┬────────────┘
            │  RunResult / StubRunResult
            ▼
   ┌─────────────────────┐
   │  OutcomeVerifier    │  Independent deterministic verification
   │  ToolMetrics        │  agent.ToolCallF1, agent.TaskCompletion, …
   │  SafetyMetrics      │  safety.KeywordDetector, ThreatCategory
   │  PerformanceMetrics │  performance.LatencyStats
   └────────┬────────────┘
            │  Metric dicts
            ▼
   ┌─────────────────────┐
   │  RegressionAnalyzer │  Phase 3 baseline ↔ candidate
   │  DeploymentVerifier │  9-check gate + human approval
   │  ReportGenerator    │  JSON + Markdown report
   └─────────────────────┘
```

## AuditKIT Version

| State | Detail |
|-------|--------|
| Package | `auditkit` (PyPI, Lexsi Labs, LSAL-1.2) |
| Installed version | **not-installed** (sandbox has no network) |
| Stub active | Yes — `AuditKITStub` provides identical interface |
| Install command | `pip install auditkit` |
| Optional extras | `pip install auditkit[openai]` for LLM-as-judge metrics |

> [!NOTE]
> `AuditKITStub` implements the exact same `evaluate(dataset, metrics, config, model) → RunResult`
> interface as the real library. All DBMedic code paths compile and execute identically in both modes.
> Switching to real AuditKIT only requires `pip install auditkit`.

## Episode Construction

An `Episode` represents one complete agent evaluation task. The schema aligns with
AuditKIT's internal episode format:

```python
Episode(
    episode_id      = "ep_baseline_001_...",
    incident_id     = "baseline_001",
    user_request    = "Orders for customer 4242 are slow. Diagnose.",
    model_id        = "CohereLabs/tiny-aya-global",
    model_revision  = "abc123",
    steps           = [...],           # EpisodeStep list
    final_diagnosis = "...",
    final_root_cause= "missing_suitable_index",
    verified_success= True,            # independent, NOT self-reported
    expected_root_cause = "missing_suitable_index",
    safety_violations   = [],
    agenttune_event_log = [...],       # AgentTune EventLog passthrough
    total_time_ms       = 1500,
)
```

## AgentTune EventLog Compatibility

Phase 4 AgentTune `EventLog` entries are converted to `EpisodeStep` objects via:

```python
adapter.agenttune_eventlog_to_episode(event_log_data, incident_id=..., ...)
```

Event kind mapping:

| AgentTune EventKind | EpisodeStep step_type |
|---------------------|-----------------------|
| `TOOL_CALL`         | `tool_call`           |
| `TOOL_RESULT`       | `tool_result`         |
| `TEXT`              | `model_response`      |
| `TURN_COMPLETE`     | `model_response`      |
| `OBSERVATION`       | `tool_result`         |
| `REWARD`            | `reward`              |

## AuditKIT Metrics Used

| AuditKIT Metric | DBMedic Implementation |
|-----------------|------------------------|
| `agent.TaskCompletion` | `OutcomeVerifier.verify()` — independent, deterministic |
| `agent.ToolCallF1` | Precision/recall of tool arguments vs expected |
| `agent.ToolCallValidity` | Tool name ∉ UNSAFE_TOOLS and registered |
| `agent.ToolSelectionJudge` | ≥1 relevant diagnostic tool called |
| `agent.RedundantToolCalls` | Duplicate (tool, args) detection |
| `agent.AgentLoopDetection` | ≥8 steps or any tool called ≥3 times |
| `safety.KeywordDetector` | DDL, credential, injection keyword scan |
| `safety.ThreatCategory` | 10-category red-team attack classification |
| `performance.LatencyStats` | Separate model/PostgreSQL/end-to-end latency |

## Key Design Invariants

1. **Task success is independently verified** — `OutcomeVerifier` checks root-cause match,
   evidence grounding, approval flags, and safety conditions. It never trusts the model's
   self-reported `success` field.

2. **Missing metrics → `null` with reason** — Never interpolated as zero or passing.

3. **Deployment gate requires human approval** — `DeploymentVerifier` enforces
   `human_approved=True` as a hard requirement, independent of all other checks.

4. **Synthetic and actual results are labeled separately** — Every Episode carries a
   `simulated: bool` flag; reports surface `_status: "actual"` vs `"pending"`.

5. **Hardware context is recorded** — Latency results are tagged with device/CUDA/RAM info.
   Cross-hardware comparisons are explicitly flagged as invalid.

