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

## Project Structure

```
DBMedic/
├── agent/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── agent.py                 # DBMedicAgent public interface
│   │   ├── orchestrator.py          # Agent loop, ReAct step execution, trajectory tracking
│   │   ├── model_client.py          # Hugging Face Transformers client for Tiny Aya Global
│   │   ├── tool_registry.py         # Registry of 8 approved read-only Phase 2 tools
│   │   ├── tool_validator.py        # Strict schema, rate-limit, and SQL-injection validation
│   │   ├── response_parser.py       # JSON extraction, schema validation, safety scanner
│   │   └── diagnosis_engine.py      # Deterministic heuristic diagnostic engine
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py               # 8 baseline evaluation metrics
│   │   ├── evaluator.py             # Baseline evaluation runner
│   │   ├── incident_cases.py        # Reproducible slow-query incidents dataset (10 baseline)
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
│   ├── baseline/
│   │   ├── results.json             # Per-case evaluation results
│   │   ├── trajectories.jsonl       # JSONL agent trajectories
│   │   └── summary.json             # Aggregate metrics summary
│   └── slow_query/
│       ├── baseline.json            # Baseline slow-query benchmark
│       ├── after_remediation.json   # Post-remediation benchmark
│       └── diagnostic_report.json   # Generated Phase 2 diagnostic report
├── database/
│   ├── scenarios/
│   │   └── slow_query.py            # Slow-query incident definition and benchmark
│   ├── schemas/
│   │   └── schema.sql               # E-commerce database schema
│   └── seeds/
│       └── generate_data.sql        # Deterministic synthetic data generator
├── docs/
│   ├── agent_architecture.md        # Phase 3 agent architecture & safety boundary reference
│   ├── tiny_aya_setup.md            # Setup guide for CohereLabs/tiny-aya-global
│   ├── baseline_evaluation.md       # Pre-training evaluation methodology & metrics
│   └── diagnostic_engine.md         # Diagnostic engine architecture and reference
├── scripts/
│   ├── run_agent.py                 # CLI interface for agent query, incident, & evaluation
│   ├── setup_lab.py                 # Initializes dbmedic_lab database and schema
│   ├── generate_dataset.py          # Populates synthetic customers, orders, products
│   ├── run_slow_query_benchmark.py  # Runs baseline / remediation benchmarks
│   └── run_diagnostics.py           # Executes Phase 2 diagnostic engine end-to-end
└── tests/
    ├── test_database_setup.py       # Validates lab database schema & constraints
    ├── test_slow_query.py           # Validates slow-query incident & gating
    ├── test_query_analyzer.py       # Plan parsing, metrics extraction, scan detection
    ├── test_table_monitor.py        # Table allowlist, size, and statistics tests
    ├── test_index_monitor.py        # Index inspection and safe identifier tests
    ├── test_diagnosis_engine.py     # Deterministic confidence & diagnostic report tests
    ├── test_tool_validator.py       # Validates schema, injection rejection, limits
    ├── test_tool_registry.py        # Tool registration and execution isolation tests
    ├── test_orchestrator.py         # Multi-step loop, max step enforcement, failure handling
    ├── test_model_client.py         # Model client loading, token protection, formatting
    ├── test_agent_integration.py    # End-to-end agent orchestration tests
    └── test_baseline_evaluation.py  # Metrics calculation, dataset, & trajectory logging
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

## Running Tests

Run unit tests (no database connection required):
```bash
python -m pytest -m "unit"
```

Run live PostgreSQL integration tests (requires Docker container active):
```bash
python -m pytest tests/ -v
```
