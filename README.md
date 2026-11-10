# DBMedic

DBMedic is an AI-powered autonomous database diagnostic agent built to detect, analyze, and resolve PostgreSQL performance bottlenecks and operational issues.

## Objectives

- **Autonomous Diagnostics**: Automatically inspect database health, query execution plans, lock contention, and resource utilization.
- **Root-Cause Analysis**: Identify underlying schema, indexing, or query anomalies using deterministic tools and domain-tuned language models.
- **Reproducible Evaluation**: Benchmark diagnostic accuracy and remediation safety against standardized database failure scenarios.

## Architecture

- **Phase 1 — Testing Environment**: PostgreSQL 17 in Docker, control database (`dbmedic`), lab database (`dbmedic_lab`), synthetic e-commerce data (200k orders), and controlled slow-query benchmark.
- **Phase 2 — Diagnostic Engine**: Deterministic PostgreSQL diagnostic tools (`query_analyzer`, `table_monitor`, `index_monitor`, `diagnosis_engine`) that investigate incidents and produce structured JSON diagnostic reports.
- **Phase 3 — Autonomous Agent (Upcoming)**: Integration with Tiny Aya Global for tool-use and reasoned troubleshooting.

## Project Structure

```
DBMedic/
├── agent/
│   ├── core/
│   │   ├── __init__.py
│   │   └── diagnosis_engine.py      # Synthesizes evidence into structured diagnosis
│   └── tools/
│       ├── __init__.py              # Read-only database connection management
│       ├── query_analyzer.py        # Plan retrieval, metrics extraction, scan detection
│       ├── table_monitor.py         # Table size, row estimates, sequential scan stats
│       └── index_monitor.py         # Index definitions, column coverage, usage metrics
├── benchmarks/
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
│   └── diagnostic_engine.md         # Diagnostic engine architecture and reference
├── scripts/
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
    └── test_diagnosis_engine.py     # Deterministic confidence & diagnostic report tests
```

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Python 3.11+
- `psycopg[binary]`, `pytest`, `python-dotenv`

### Setup

1. Clone the repository and navigate to the project root:
   ```bash
   cd DBMedic
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create your local environment configuration from the example template:
   ```bash
   cp .env.example .env
   ```
   Ensure `DATABASE_URL` and `LAB_DATABASE_URL` point to your PostgreSQL instance.

3. Start PostgreSQL 17:
   ```bash
   docker compose up -d
   ```

4. Initialize the lab database and synthetic dataset (Phase 1):
   ```bash
   python scripts/setup_lab.py
   python scripts/generate_dataset.py
   ```

## Running the Diagnostic Engine (Phase 2)

Execute the end-to-end diagnostic runner to analyze the slow-query incident on `dbmedic_lab`:

```bash
python scripts/run_diagnostics.py
```

This generates a standardized report at `benchmarks/slow_query/diagnostic_report.json`:

```json
{
  "incident_id": "slow_query_001",
  "incident_type": "slow_query",
  "status": "diagnosed",
  "severity": "high",
  "summary": "Query performance degradation detected",
  "root_cause": {
    "type": "missing_suitable_index",
    "description": "Sequential scan detected on orders"
  },
  "evidence": {
    "scan_type": "Sequential Scan",
    "execution_time_ms": 18.42,
    "table_name": "orders",
    "filter_column": "customer_id",
    "suitable_index_exists": false
  },
  "recommendation": {
    "action": "create_index",
    "sql": "CREATE INDEX idx_orders_customer_id ON orders(customer_id)",
    "requires_approval": true
  },
  "confidence": 0.95
}
```

## Running Tests

Run the full test suite with pytest:

```bash
python -m pytest tests/ -v
```
