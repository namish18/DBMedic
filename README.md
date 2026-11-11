# DBMedic

DBMedic is an AI-powered autonomous database diagnostic and self-healing agent built to detect, analyze, and resolve PostgreSQL performance bottlenecks and operational issues safely.

## Objectives

- **Autonomous Diagnostics**: Automatically inspect database health, query execution plans, lock contention, and resource utilization.
- **Root-Cause Analysis**: Identify underlying schema, indexing, or query anomalies using deterministic tools and domain-tuned language models.
- **Reproducible Evaluation**: Benchmark diagnostic accuracy and remediation safety against standardized database failure scenarios.
- **Safety First**: Autonomous read-only diagnostics with mandatory human approval for database mutations (DDL/remediation).

## Architecture

- **Phase 1 — Testing Environment**: PostgreSQL 17 in Docker, control database (`dbmedic`), lab database (`dbmedic_lab`), synthetic e-commerce data (200k orders), and controlled slow-query benchmark.
- **Phase 2 — Diagnostic Engine**: Deterministic PostgreSQL diagnostic tools (`query_analyzer`, `table_monitor`, `index_monitor`, `diagnosis_engine`) that investigate incidents and produce structured JSON diagnostic reports.
- **Phase 3 — Autonomous Agent Integration & Baseline Evaluation**: Full ReAct agent integration with `CohereLabs/tiny-aya-global`, strict tool-calling protocol, argument validation, trajectory logging, and standardized baseline evaluation.
- **Phase 4 — AgentTune Training, Failure Learning & Self-Healing Pipeline**: 10-category failure classifier, verified Phase 2 correction synthesis, leakage-free Tiny Aya SFT dataset curation with SHA-256 verification, AgentTune LoRA SFT distillation, gated GRPO experimentation, deterministic diagnostic rewards, and human-approved deployment gating.
- **Phase 5 — AuditKIT Integration, Agent Evaluation & Deployment Verification**: Independent AuditKIT evaluation spine (Lexsi Labs, LSAL-1.2) with offline stub fallback, DBMedic episode adapter, 18-scenario evaluation dataset, deterministic 10-category red-team benchmark (10/10 attacks blocked), regression analyzer, and 9-check deployment gate with mandatory human approval.
- **Phase 6 — Final Integration, End-to-End Validation & Hackathon Demonstration**: Unified `scripts/dbmedic.py` CLI (`doctor`, `diagnose`, `approve`, `verify`, `evaluate`, `report`), environment readiness checker, strict incident lifecycle state machine (`CREATED -> INVESTIGATING -> DIAGNOSED -> AWAITING_APPROVAL -> APPROVED -> REMEDIATING -> VERIFYING -> RESOLVED`), human-only approval gate with SQL SHA-256 & target DB identity binding, allowlisted remediation wrapper, before/after plan & benchmark verifier, append-only SHA-256 chained audit trail, reproducible `demo/` scenarios, and optional Streamlit dashboard (`dashboard/app.py`).

## Project Structure

```
DBMedic/
├── agent/
│   ├── audit/                       # Phase 5: AuditKIT adapter, metrics, verifier & regression analyzer
│   │   ├── __init__.py
│   │   ├── auditkit_adapter.py      # Trajectory -> Episode adapter & AuditKITStub fallback
│   │   ├── episode_builder.py       # Multi-source episode builder
│   │   ├── outcome_verifier.py      # Deterministic outcome & plan alignment verifier
│   │   ├── performance_metrics.py   # Latency & throughput metrics
│   │   ├── regression_analyzer.py   # Base vs Candidate regression checks
│   │   ├── report_generator.py      # Evaluation JSON & Markdown report generation
│   │   ├── safety_metrics.py        # Keyword & threat-category detectors
│   │   └── tool_metrics.py          # ToolCallF1, validity, selection judge, loop detection
│   ├── core/
│   │   ├── __init__.py
│   │   ├── agent.py                 # DBMedicAgent public interface
│   │   ├── approval_manager.py      # Phase 6: Human-only SQL approval & SHA-256 verification
│   │   ├── audit_logger.py          # Phase 6: Append-only JSONL SHA-256 chained audit log
│   │   ├── diagnosis_engine.py      # Deterministic heuristic diagnostic engine
│   │   ├── environment_check.py     # Phase 6: Python, PostgreSQL, HF, Tiny Aya, AgentTune & AuditKIT checks
│   │   ├── incident_orchestrator.py # Phase 6: Strict incident lifecycle state machine
│   │   ├── model_client.py          # Hugging Face Transformers client for Tiny Aya Global
│   │   ├── orchestrator.py          # Agent loop, ReAct step execution, trajectory tracking
│   │   ├── remediation_manager.py   # Phase 6: Allowlisted safe remediation wrapper over Phase 1 DDL
│   │   ├── remediation_verifier.py  # Phase 6: Before/after execution plan & latency verifier
│   │   ├── response_parser.py       # JSON extraction, schema validation, safety scanner
│   │   ├── tool_registry.py         # Registry of 8 approved read-only Phase 2 tools
│   │   └── tool_validator.py        # Strict schema, rate-limit, and SQL-injection validation
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── auditkit_evaluator.py    # Phase 5: Recorded + live AuditKIT evaluation runner
│   │   ├── auditkit_scenarios.py    # Phase 5: 18 evaluation scenarios + 5 hidden scenarios
│   │   ├── deployment_verifier.py   # Phase 5: 9-check deployment gate with human approval
│   │   ├── evaluator.py             # Baseline evaluation runner
│   │   ├── final_evaluator.py       # Phase 6: Unified Phases 3-6 metrics & consolidated reports
│   │   ├── incident_cases.py        # Reproducible slow-query incidents dataset (10 baseline)
│   │   ├── metrics.py               # 8 baseline evaluation metrics
│   │   ├── training_comparison.py   # Phase 4: Base vs Candidate comparison & gate
│   │   └── trajectory_logger.py     # Trajectory recorder for AgentTune training
│   ├── prompts/
│   │   ├── system_prompt.txt        # DBMedic agent system prompt & tool protocol
│   │   └── tool_call_prompt.txt     # Observation feedback prompt
│   └── tools/
│       ├── __init__.py              # Read-only database connection management
│       ├── query_analyzer.py        # Plan retrieval, metrics extraction, scan detection
│       ├── table_monitor.py         # Table size, row estimates, sequential scan stats
│       └── index_monitor.py         # Index definitions, column coverage, usage metrics
├── benchmarks/
│   ├── auditkit/                    # Phase 5 AuditKIT recorded/redteam/comparison/report artifacts
│   ├── baseline/                    # Preserved Phase 3 baseline results & trajectories.jsonl
│   ├── final_demo/                  # Phase 6 remediation_comparison.json & final_metrics.json
│   └── slow_query/                  # Preserved Phase 1 baseline, after_remediation & diagnostic report
├── dashboard/                       # Optional Phase 6 Streamlit dashboard (CLI remains primary)
│   ├── app.py
│   └── components/                  # incident, diagnostic, approval, benchmark & audit panels
├── database/
│   ├── scenarios/
│   │   └── slow_query.py            # Slow-query incident definition and benchmark
│   ├── schemas/
│   │   └── schema.sql               # E-commerce database schema
│   └── seeds/
│       └── generate_data.sql        # Deterministic synthetic data generator
├── demo/                            # Phase 6 reproducible hackathon demonstration suite
│   ├── scenarios/                   # slow_query_{missing_index,existing_index,selective_filter}.json
│   ├── expected_results/            # Ground-truth validation specs (isolated from model prompts)
│   └── scripts/                     # reset_demo_database.py, run_complete_demo.py
├── docs/
│   ├── agent_architecture.md        # Phase 3 agent architecture & safety boundary reference
│   ├── agenttune_integration.md     # Phase 4 AgentTune spine, EventLog & TRL compatibility
│   ├── auditkit_integration.md      # Phase 5 AuditKIT adapter & stub specification
│   ├── baseline_evaluation.md       # Pre-training evaluation methodology & metrics
│   ├── deployment_guide.md          # Phase 6 setup, verification & reproduction guide
│   ├── deployment_verification.md   # Phase 5 9-check deployment gate specification
│   ├── diagnostic_engine.md         # Diagnostic engine architecture and reference
│   ├── end_to_end_workflow.md       # Phase 6 incident lifecycle state machine & CLI flow
│   ├── evaluation_methodology.md    # Phase 5 evaluation protocol
│   ├── failure_learning.md          # 10-category failure taxonomy, repairs & self-healing loop
│   ├── final_architecture.md        # Phase 6 complete system architecture across Phases 1-6
│   ├── hackathon_demo_script.md     # 5-7 minute live hackathon presentation script
│   ├── reward_design.md             # Configurable deterministic rewards & anti-hacking rules
│   ├── safety_and_approval.md       # Phase 6 human-only approval gate & audit threat model
│   ├── safety_evaluation.md         # Phase 5 deterministic 10-category red-team suite
│   ├── tiny_aya_setup.md            # Setup guide for CohereLabs/tiny-aya-global
│   └── training_pipeline.md         # AgentTune SFT distillation & gated GRPO guide
├── reports/                         # Phase 6 generated runtime & summary artifacts
│   ├── environment_readiness.json   # Output of `python scripts/dbmedic.py doctor`
│   ├── audit_trail.jsonl            # Append-only SHA-256 chained event trail
│   ├── final_summary.json           # Output of `python scripts/dbmedic.py report`
│   └── final_summary.md             # Human-readable consolidated summary report
├── scripts/
│   ├── dbmedic.py                   # Phase 6 Unified CLI (doctor, diagnose, approve, verify, evaluate, report)
│   ├── run_agent.py                 # CLI interface for agent query, incident, & evaluation
│   ├── setup_lab.py                 # Initializes dbmedic_lab database and schema
│   ├── generate_dataset.py          # Populates synthetic customers, orders, products
│   ├── run_slow_query_benchmark.py  # Runs baseline / remediation benchmarks
│   ├── run_diagnostics.py           # Executes Phase 2 diagnostic engine end-to-end
│   ├── collect_training_trajectories.py # Collects real or controlled failure trajectories
│   ├── analyze_failures.py          # Runs 10-category failure analysis on trajectories
│   ├── build_training_dataset.py    # Curates train/validation/hidden_test + SHA-256 manifest
│   ├── validate_training_dataset.py # Audits splits for zero leakage, safety & protocol
│   ├── train_agent.py               # Runs preflight, SFT distillation, or gated GRPO
│   ├── evaluate_trained_agent.py    # Deterministic post-training checkpoint evaluation
│   ├── compare_models.py            # Base vs candidate 8-metric comparison & gate
│   ├── run_agent_redteam.py         # Phase 5 deterministic 10-category red-team suite
│   ├── run_auditkit_evaluation.py   # Phase 5 AuditKIT recorded / live evaluation CLI
│   ├── compare_audit_results.py     # Phase 5 base vs candidate comparison & report CLI
│   └── verify_deployment.py         # Phase 5 9-check deployment gate CLI
├── training/
│   ├── configs/                     # dataset_config.yaml, sft_config.yaml, grpo_config.yaml
│   ├── failure_analysis/            # 10-category classifier, batch analyzer & verifier
│   ├── rewards/                     # Deterministic reward_config.yaml & safety/diagnostic rewards
│   ├── data/                        # Tiny Aya strict alternation formatter, curator, validator & splits
│   ├── agenttune/                   # EventLog/Project adapter, LoRA SFT distillation, gated GRPO
│   └── experiments/                 # Baseline snapshot & SFT/GRPO status/provenance artifacts
└── tests/                           # 307 total tests across Phases 1-6 (271 unit, 35 integration, 1 model)
    ├── test_database_setup.py       # Phase 1: lab database schema & constraints
    ├── test_slow_query.py           # Phase 1: slow-query incident & gating
    ├── test_query_analyzer.py       # Phase 2: plan parsing, metrics extraction, scan detection
    ├── test_table_monitor.py        # Phase 2: table allowlist, size, and statistics
    ├── test_index_monitor.py        # Phase 2: index inspection and safe identifiers
    ├── test_diagnosis_engine.py     # Phase 2: deterministic heuristic scores & report
    ├── test_tool_validator.py       # Phase 3: schema, injection rejection, rate limits
    ├── test_tool_registry.py        # Phase 3: tool registration and execution isolation
    ├── test_orchestrator.py         # Phase 3: multi-step loop & max step enforcement
    ├── test_model_client.py         # Phase 3: model client loading, token protection
    ├── test_agent_integration.py    # Phase 3: end-to-end agent orchestration
    ├── test_baseline_evaluation.py  # Phase 3: metrics, incident cases, trajectory logging
    ├── test_failure_classifier.py   # Phase 4: all 10 failure categories & tolerance rules
    ├── test_correction_generator.py # Phase 4: deterministic correction & verification
    ├── test_trajectory_adapter.py   # Phase 4: AgentTune EventLog, Project & DictToolHarness
    ├── test_training_dataset.py     # Phase 4: curation, dedup, zero leakage & -100 masking
    ├── test_reward_functions.py     # Phase 4: deterministic reward weights & anti-hacking
    ├── test_training_config.py      # Phase 4: YAML configs, LoRA, preflight & GRPO gating
    ├── test_model_comparison.py     # Phase 4: 8-metric A/B comparison & self-healing loop
    ├── test_auditkit_adapter.py     # Phase 5: AuditKIT adapter & AuditKITStub
    ├── test_episode_builder.py      # Phase 5: episode conversion & simulation separation
    ├── test_outcome_verifier.py     # Phase 5: deterministic outcome verification
    ├── test_auditkit_evaluator.py   # Phase 5: recorded/live evaluation runner
    ├── test_safety_metrics.py       # Phase 5: keyword/threat detection & red-team checks
    ├── test_regression_analyzer.py  # Phase 5: regression detection & null handling
    ├── test_deployment_verifier.py  # Phase 5: 9-check deployment gate & human approval
    ├── test_environment_check.py    # Phase 6: readiness checks & credential redaction
    ├── test_incident_orchestrator.py# Phase 6: lifecycle transitions & zero silent fallback
    ├── test_approval_manager.py     # Phase 6: anti-spoofing, SQL SHA-256, expiry & DB lock
    ├── test_remediation_manager.py  # Phase 6: allowlist, idempotency & no auto-retry
    ├── test_remediation_verifier.py # Phase 6: plan comparison & small-table seq scan handling
    ├── test_audit_logger.py         # Phase 6: append-only JSONL hash chain & secret redaction
    └── test_final_evaluation.py     # Phase 6: unified metrics, CLI exit codes & demo flow
```

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Python 3.11+
- `psycopg[binary]`, `pytest`, `python-dotenv`
- (Optional for Phase 3 model inference): `transformers`, `torch`, `accelerate`, `huggingface_hub`

### Setup

1. Clone the repository and navigate to the project root:
   ```bash
   cd DBMedic
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure your local `.env`:
   ```bash
   cp .env.example .env
   ```
   Ensure `LAB_DATABASE_URL` is configured, and set `HF_TOKEN` if running real model inference.

4. Start PostgreSQL 17:
   ```bash
   docker compose up -d
   ```

5. Initialize the lab database and synthetic dataset:
   ```bash
   python scripts/setup_lab.py
   python scripts/generate_dataset.py
   ```

## Running the Agent (Phase 3)

The DBMedic agent can be executed via `scripts/run_agent.py`:

```bash
# 1. Ask a free-text database diagnostic question
python scripts/run_agent.py --query "Why is my orders query taking so long?"

# 2. Run an investigation on a registered incident
python scripts/run_agent.py --incident slow_query_001

# 3. List all known incidents and evaluation dataset cases
python scripts/run_agent.py --list-incidents

# 4. Run baseline evaluation
python scripts/run_agent.py --evaluate --max-cases 3
```

## Phase 4 — AgentTune Training & Failure Learning

```powershell
# 1. Collect trajectories (or generate a clearly tagged simulated failure sample offline)
python scripts/collect_training_trajectories.py --generate-controlled-failures-sample

# 2. Analyze failures across all 10 failure categories
python scripts/analyze_failures.py

# 3. Curate leakage-free splits (train / validation / sanitized hidden_test) & SHA-256 manifest
python scripts/build_training_dataset.py
python scripts/validate_training_dataset.py

# 4. Run readiness preflight, primary AgentTune SFT (or dry-run), and gated GRPO checks
python scripts/train_agent.py --mode preflight
python scripts/train_agent.py --mode sft --dry-run
python scripts/train_agent.py --mode grpo --dry-run

# 5. Evaluate checkpoint deterministically and compare Base vs Candidate
python scripts/evaluate_trained_agent.py --split validation --dry-run
python scripts/compare_models.py --dry-run
```

## Phase 5 — AuditKIT Evaluation & Red-Team Verification

```powershell
# 1. Deterministic 10-category red-team suite (offline)
python scripts/run_agent_redteam.py

# 2. Recorded-mode AuditKIT evaluation (uses AuditKITStub when auditkit package is absent)
python scripts/run_auditkit_evaluation.py --mode recorded

# 3. Regression analysis & 9-check deployment gate
python scripts/compare_audit_results.py
python scripts/verify_deployment.py
```

## Phase 6 — Unified DBMedic Application & End-to-End Demo

Run the entire DBMedic workflow through one entry point (`scripts/dbmedic.py`):

```powershell
# 1. Verify environment readiness (PostgreSQL, Tiny Aya, AgentTune, AuditKIT/Stub)
python scripts/dbmedic.py doctor

# 2. Reset dbmedic_lab to reproducible missing-index baseline (when PostgreSQL is active)
python demo/scripts/reset_demo_database.py

# 3. Diagnose incident (read-only; stops at AWAITING_APPROVAL — no DDL executed)
python scripts/dbmedic.py diagnose --incident slow_query_001 --mode deterministic
# Or live Tiny Aya mode when HF_TOKEN + GPU/RAM are available (never falls back silently):
# python scripts/dbmedic.py diagnose --incident slow_query_001 --mode tiny-aya

# 4. Explicit human operator approval (blocks model actors & raw booleans)
python scripts/dbmedic.py approve --incident slow_query_001

# 5. Execute allowlisted DDL & verify real execution plan + 7-run benchmark speedup
python scripts/dbmedic.py verify --incident slow_query_001

# 6. Connect Phase 3-6 evaluations and generate consolidated summary + audit check
python scripts/dbmedic.py evaluate
python scripts/dbmedic.py report

# Optional visual dashboard (CLI remains the primary supported interface):
streamlit run dashboard/app.py
```

### Offline Demonstration Mode

When PostgreSQL Docker or `HF_TOKEN`/GPU is unavailable, run DBMedic's explicitly tagged synthetic demo (`simulated=true`):

```powershell
python demo/scripts/run_complete_demo.py --scenario slow_query_missing_index --mode simulated
```

## Known Limitations

- **Live Tiny Aya Inference & Fine-Tuning**: Requires a valid `HF_TOKEN` with access to `CohereLabs/tiny-aya-global` and an NVIDIA CUDA GPU (`>= 8 GB` VRAM for SFT / `>= 12 GB` for GRPO). Missing live metrics remain `null` and are never fabricated.
- **AuditKIT Stub Disclosure**: When the external `auditkit` package (Lexsi Labs, LSAL-1.2 non-commercial/source-available license) is not installed, DBMedic explicitly activates and labels `AuditKITStub` rather than claiming independent AuditKIT validation.
- **Audit Trail Threat Model**: `reports/audit_trail.jsonl` uses local SHA-256 hash chaining to detect accidental corruption or single-line tampering; production deployments should forward events to external WORM/KMS-signed storage.

## Running Tests

Run all offline unit tests across Phases 1–6 (271 tests; no database or GPU required):
```bash
python -m pytest -m "unit" -v
```

Run the full test suite including live PostgreSQL integration tests (requires Docker container active):
```bash
python -m pytest tests/ -v
```

## Additional Documentation

- [Final System Architecture](docs/final_architecture.md)
- [End-to-End Incident Workflow](docs/end_to_end_workflow.md)
- [Safety, Human Approval & Audit Model](docs/safety_and_approval.md)
- [Deployment & Reproduction Guide](docs/deployment_guide.md)
- [Hackathon Live Demo Script (5-7 min)](docs/hackathon_demo_script.md)

