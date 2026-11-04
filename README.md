# DBMedic

DBMedic is an AI-powered autonomous database diagnostic agent built to detect, analyze, and resolve PostgreSQL performance bottlenecks and operational issues.

## Objectives

- **Autonomous Diagnostics**: Automatically inspect database health, query execution plans, lock contention, and resource utilization.
- **Root-Cause Analysis**: Identify underlying schema, indexing, or query anomalies using domain-tuned language models.
- **Reproducible Evaluation**: Benchmark diagnostic accuracy and remediation safety against standardized database failure scenarios.

## Planned Technologies

- **Language Model**: Tiny Aya Global for lightweight, reasoning-driven diagnostic inference.
- **Agent Fine-Tuning**: AgentTune for specializing agent tool-use and multi-step troubleshooting trajectories.
- **Evaluation Framework**: AuditKIT for systematic auditing, safety verification, and benchmark scoring.
- **Database**: PostgreSQL 17 (containerized via Docker Compose).
- **Backend & Agent Runtime**: Python.

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Python 3.11+

### Setup

1. Clone the repository and navigate to the project root:
   ```bash
   cd DBMedic
   ```

2. Create your local environment configuration from the example template:
   ```bash
   cp .env.example .env
   ```
   Update `POSTGRES_PASSWORD` and `DATABASE_URL` in `.env` with your local development credentials.

3. Start the PostgreSQL 17 service:
   ```bash
   docker compose up -d
   ```

4. Verify the database container is healthy:
   ```bash
   docker compose ps
   ```
