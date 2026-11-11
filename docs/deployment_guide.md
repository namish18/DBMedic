# DBMedic Deployment & Reproduction Guide

## 1. Prerequisites

- Windows / Linux / macOS with **Python 3.11+** (verified on Python 3.12.10)
- **Docker & Docker Compose** (for PostgreSQL 17 lab database `dbmedic_lab`)
- **Optional for Live Tiny Aya & Fine-Tuning**:
  - Valid `HF_TOKEN` with access granted to `CohereLabs/tiny-aya-global`
  - NVIDIA CUDA GPU (>= 8 GB VRAM for inference/SFT, >= 12 GB VRAM for GRPO)
- **Optional for Real AuditKIT Validation**:
  - `pip install auditkit` (note: LSAL-1.2 license is non-commercial/source-available; review before commercial deployment)

## 2. Fresh Setup (Windows PowerShell)

```powershell
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure .env from example (do NOT commit .env)
Copy-Item .env.example .env

# 3. Start PostgreSQL 17 container and initialize dbmedic_lab
docker compose up -d
python scripts/setup_lab.py
python scripts/generate_dataset.py

# 4. Run doctor check
python scripts/dbmedic.py doctor
```

## 3. Running End-to-End Incident Lifecycle

```powershell
# Reset baseline slow-query state
python demo/scripts/reset_demo_database.py

# Diagnose -> Approve -> Verify
python scripts/dbmedic.py diagnose --incident slow_query_001 --mode deterministic
python scripts/dbmedic.py approve --incident slow_query_001
python scripts/dbmedic.py verify --incident slow_query_001

# Unified Evaluation & Summary Report
python scripts/dbmedic.py evaluate
python scripts/dbmedic.py report

# Optional Visual Dashboard
streamlit run dashboard/app.py
```

## 4. Test Verification Matrix

```powershell
# Offline unit tests (Phases 1–6, no DB or GPU required)
python -m pytest -m "unit" -v

# Full test suite including live PostgreSQL integration tests (requires docker compose up -d)
python -m pytest tests/ -v
```

