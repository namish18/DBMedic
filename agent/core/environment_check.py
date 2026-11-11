"""
agent/core/environment_check.py
================================
Unified environment readiness and dependency inspection for DBMedic Phase 6.

Checks:
  1. Python runtime, required & optional packages, and .env variable presence
  2. Docker CLI & PostgreSQL connection (dbmedic_lab)
  3. dbmedic_lab schema & required tables (customers, products, orders, order_items)
  4. Hugging Face Hub authentication & cache status
  5. CohereLabs/tiny-aya-global model & checkpoint availability
  6. AgentTune package version & curated dataset status
  7. AuditKIT real package vs AuditKITStub distinction
  8. GPU / CUDA & system RAM availability

Generates ``reports/environment_readiness.json`` with explicit ``READY``,
``BLOCKED``, and ``OPTIONAL`` component statuses without ever exposing
passwords, tokens, or raw DSNs.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import logging
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from dotenv import dotenv_values, load_dotenv

from agent.core.audit_logger import redact_database_url, redact_secrets

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_READINESS_REPORT_PATH = PROJECT_ROOT / "reports" / "environment_readiness.json"
DEFAULT_MODEL_ID = "CohereLabs/tiny-aya-global"
EXPECTED_LAB_TABLES: tuple[str, ...] = ("customers", "products", "orders", "order_items")

REQUIRED_ENV_VARS: tuple[str, ...] = ("LAB_DATABASE_URL",)
RECOMMENDED_ENV_VARS: tuple[str, ...] = (
    "DATABASE_URL",
    "POSTGRES_DB",
    "POSTGRES_USER",
    "POSTGRES_PASSWORD",
    "POSTGRES_HOST",
    "POSTGRES_PORT",
)
OPTIONAL_MODEL_ENV_VARS: tuple[str, ...] = ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN")

STATUS_READY = "READY"
STATUS_BLOCKED = "BLOCKED"
STATUS_OPTIONAL = "OPTIONAL"


def _load_env_mapping(env_overrides: dict[str, str] | None = None) -> dict[str, str]:
    """Load effective env vars without mutating process env if overrides passed."""
    if env_overrides is not None:
        return dict(env_overrides)
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        load_dotenv(env_file, override=False)
    merged: dict[str, str] = {}
    if env_file.exists():
        for k, v in dotenv_values(env_file).items():
            if v is not None:
                merged[k] = v
    for k, v in os.environ.items():
        merged[k] = v
    return merged


def _package_info(dist_name: str, import_name: str | None = None) -> dict[str, Any]:
    """Inspect installation state and version for one Python package."""
    mod_name = import_name or dist_name
    spec = importlib.util.find_spec(mod_name)
    if spec is None:
        return {"installed": False, "version": None}

    version: str | None = None
    try:
        version = importlib.metadata.version(dist_name)
    except Exception:
        try:
            mod = importlib.import_module(mod_name)
            version = getattr(mod, "__version__", "installed")
        except Exception:
            version = "installed"
    return {"installed": True, "version": str(version)}


def parse_database_identity(url: str | None) -> dict[str, Any]:
    """Extract sanitized database identity (host, port, database) with zero secrets."""
    if not url or not str(url).strip():
        return {
            "configured": False,
            "host": None,
            "port": None,
            "database": None,
            "redacted_url": None,
        }
    parsed = urlparse(str(url).strip())
    db_name = parsed.path.lstrip("/") if parsed.path else None
    return {
        "configured": True,
        "host": parsed.hostname,
        "port": parsed.port or 5432,
        "database": db_name or None,
        "redacted_url": redact_database_url(str(url).strip()),
    }


def check_python_environment(
    *,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Check Python version, required/optional packages, and env configuration."""
    effective_env = _load_env_mapping(env)
    import sys

    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    py_ok = sys.version_info >= (3, 11)

    core_packages = {
        "psycopg": _package_info("psycopg"),
        "python-dotenv": _package_info("python-dotenv", "dotenv"),
        "pytest": _package_info("pytest"),
        "pyyaml": _package_info("PyYAML", "yaml"),
        "psutil": _package_info("psutil"),
    }
    optional_packages = {
        "torch": _package_info("torch"),
        "transformers": _package_info("transformers"),
        "accelerate": _package_info("accelerate"),
        "huggingface_hub": _package_info("huggingface_hub"),
        "peft": _package_info("peft"),
        "trl": _package_info("trl"),
        "datasets": _package_info("datasets"),
        "agenttune": _package_info("agenttune"),
        "auditkit": _package_info("auditkit"),
        "streamlit": _package_info("streamlit"),
    }

    missing_core_pkgs = [k for k, v in core_packages.items() if not v["installed"]]
    missing_required_env = [
        k for k in REQUIRED_ENV_VARS if not str(effective_env.get(k, "")).strip()
    ]
    missing_recommended_env = [
        k for k in RECOMMENDED_ENV_VARS if not str(effective_env.get(k, "")).strip()
    ]
    env_presence = {
        k: bool(str(effective_env.get(k, "")).strip())
        for k in (REQUIRED_ENV_VARS + RECOMMENDED_ENV_VARS + OPTIONAL_MODEL_ENV_VARS)
    }

    blockers: list[str] = []
    if not py_ok:
        blockers.append(f"Python >= 3.11 recommended, found {py_ver}.")
    if missing_core_pkgs:
        blockers.append(f"Missing required Python packages: {', '.join(missing_core_pkgs)}")
    if missing_required_env:
        blockers.append(
            f"Missing required environment variables: {', '.join(missing_required_env)}"
        )

    status = STATUS_READY if not blockers else STATUS_BLOCKED
    return {
        "component": "python_environment",
        "status": status,
        "python_version": py_ver,
        "python_supported": py_ok,
        "env_file_exists": (PROJECT_ROOT / ".env").exists(),
        "env_vars_present": env_presence,
        "missing_required_env": missing_required_env,
        "missing_recommended_env": missing_recommended_env,
        "core_packages": core_packages,
        "optional_packages": optional_packages,
        "blockers": blockers,
    }


def _check_docker_status() -> dict[str, Any]:
    """Detect whether the Docker CLI is installed and postgres container is running."""
    docker_bin = shutil.which("docker")
    if not docker_bin:
        return {
            "docker_cli_available": False,
            "docker_running": False,
            "dbmedic_container_running": False,
            "detail": "docker CLI not found on PATH",
        }

    try:
        proc = subprocess.run(
            [docker_bin, "ps", "--format", "{{.Names}}|{{.Status}}"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if proc.returncode != 0:
            return {
                "docker_cli_available": True,
                "docker_running": False,
                "dbmedic_container_running": False,
                "detail": (proc.stderr or "docker daemon unreachable").strip()[:200],
            }
        lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
        dbmedic_running = any("dbmedic" in line.lower() for line in lines)
        return {
            "docker_cli_available": True,
            "docker_running": True,
            "dbmedic_container_running": dbmedic_running,
            "containers": lines[:10],
            "detail": "Docker daemon reachable",
        }
    except Exception as exc:
        return {
            "docker_cli_available": True,
            "docker_running": False,
            "dbmedic_container_running": False,
            "detail": f"Docker check failed: {type(exc).__name__}",
        }


def check_database_connection(
    *,
    timeout_sec: int = 2,
    env: dict[str, str] | None = None,
    conn_factory: Any = None,
) -> dict[str, Any]:
    """Verify Docker status and read-only connectivity to ``dbmedic_lab``."""
    effective_env = _load_env_mapping(env)
    lab_url = effective_env.get("LAB_DATABASE_URL", "").strip()
    target_identity = parse_database_identity(lab_url)
    docker_info = _check_docker_status()

    if not lab_url:
        return {
            "component": "database_connection",
            "status": STATUS_BLOCKED,
            "connected": False,
            "target_database": target_identity,
            "docker": docker_info,
            "sandbox_socket_blocked": False,
            "server_version": None,
            "blocker": "LAB_DATABASE_URL is not configured in environment or .env.",
        }

    try:
        if conn_factory is not None:
            ctx = conn_factory()
        else:
            import psycopg

            ctx = psycopg.connect(
                lab_url,
                connect_timeout=timeout_sec,
                options="-c default_transaction_read_only=on",
            )
        with ctx as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT current_database(), current_user, inet_server_addr()::text, "
                    "inet_server_port(), version();"
                )
                row = cur.fetchone()
        return {
            "component": "database_connection",
            "status": STATUS_READY,
            "connected": True,
            "target_database": {
                **target_identity,
                "live_current_database": row[0] if row else target_identity.get("database"),
                "live_server_port": row[3] if row and row[3] else target_identity.get("port"),
            },
            "docker": docker_info,
            "sandbox_socket_blocked": False,
            "server_version": str(row[4]).split(",")[0] if row and row[4] else None,
            "blocker": None,
        }
    except Exception as exc:
        msg = " ".join(redact_secrets(str(exc)).split())
        socket_blocked = "10013" in msg or "permission denied" in msg.lower()
        return {
            "component": "database_connection",
            "status": STATUS_BLOCKED,
            "connected": False,
            "target_database": target_identity,
            "docker": docker_info,
            "sandbox_socket_blocked": socket_blocked,
            "server_version": None,
            "blocker": (
                f"PostgreSQL connection failed ({type(exc).__name__}): {msg[:220]}"
            ),
        }


def check_lab_schema(
    *,
    timeout_sec: int = 2,
    env: dict[str, str] | None = None,
    conn_factory: Any = None,
) -> dict[str, Any]:
    """Verify dbmedic_lab exists and contains all expected tables & baseline telemetry."""
    conn_status = check_database_connection(
        timeout_sec=timeout_sec,
        env=env,
        conn_factory=conn_factory,
    )
    if not conn_status["connected"]:
        return {
            "component": "lab_schema",
            "status": STATUS_BLOCKED,
            "schema_valid": False,
            "expected_tables": list(EXPECTED_LAB_TABLES),
            "present_tables": [],
            "missing_tables": list(EXPECTED_LAB_TABLES),
            "orders_estimated_rows": None,
            "idx_orders_customer_id_exists": None,
            "blocker": conn_status["blocker"],
        }

    effective_env = _load_env_mapping(env)
    lab_url = effective_env.get("LAB_DATABASE_URL", "").strip()

    try:
        if conn_factory is not None:
            ctx = conn_factory()
        else:
            import psycopg

            ctx = psycopg.connect(
                lab_url,
                connect_timeout=timeout_sec,
                options="-c default_transaction_read_only=on",
            )
        with ctx as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT tablename
                    FROM pg_tables
                    WHERE schemaname = 'public'
                      AND tablename = ANY(%s)
                    ORDER BY tablename;
                    """,
                    (list(EXPECTED_LAB_TABLES),),
                )
                present = [str(r[0]) for r in cur.fetchall()]
                missing = [t for t in EXPECTED_LAB_TABLES if t not in present]

                orders_rows: int | None = None
                idx_exists: bool | None = None
                if "orders" in present:
                    cur.execute(
                        "SELECT COALESCE(reltuples::bigint, 0) FROM pg_class c "
                        "JOIN pg_namespace n ON n.oid = c.relnamespace "
                        "WHERE n.nspname = 'public' AND c.relname = 'orders';"
                    )
                    r_est = cur.fetchone()
                    orders_rows = int(r_est[0]) if r_est else None

                    cur.execute(
                        "SELECT EXISTS (SELECT 1 FROM pg_indexes "
                        "WHERE schemaname = 'public' AND indexname = 'idx_orders_customer_id');"
                    )
                    r_idx = cur.fetchone()
                    idx_exists = bool(r_idx[0]) if r_idx else False

        schema_ok = len(missing) == 0
        return {
            "component": "lab_schema",
            "status": STATUS_READY if schema_ok else STATUS_BLOCKED,
            "schema_valid": schema_ok,
            "expected_tables": list(EXPECTED_LAB_TABLES),
            "present_tables": present,
            "missing_tables": missing,
            "orders_estimated_rows": orders_rows,
            "idx_orders_customer_id_exists": idx_exists,
            "blocker": (
                None
                if schema_ok
                else f"Missing dbmedic_lab tables: {', '.join(missing)}. Run scripts/setup_lab.py."
            ),
        }
    except Exception as exc:
        return {
            "component": "lab_schema",
            "status": STATUS_BLOCKED,
            "schema_valid": False,
            "expected_tables": list(EXPECTED_LAB_TABLES),
            "present_tables": [],
            "missing_tables": list(EXPECTED_LAB_TABLES),
            "orders_estimated_rows": None,
            "idx_orders_customer_id_exists": None,
            "blocker": f"Schema verification error: {redact_secrets(str(exc))[:240]}",
        }


def check_huggingface_access(
    *,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Check HF Hub package & authentication state without ever exposing token values."""
    effective_env = _load_env_mapping(env)
    token_var: str | None = None
    for key in OPTIONAL_MODEL_ENV_VARS:
        val = str(effective_env.get(key, "")).strip()
        if val:
            token_var = key
            break

    hf_pkg = _package_info("huggingface_hub")
    local_cache_found = False
    try:
        cache_root = Path(
            effective_env.get("HF_HOME")
            or (Path.home() / ".cache" / "huggingface" / "hub")
        )
        model_slug = f"models--{DEFAULT_MODEL_ID.replace('/', '--')}"
        if (cache_root / model_slug).exists() or (cache_root / "hub" / model_slug).exists():
            local_cache_found = True
    except Exception:
        local_cache_found = False

    token_present = token_var is not None
    if token_present and hf_pkg["installed"]:
        status = STATUS_READY
        note = f"HF token detected via {token_var} (value redacted)."
    elif local_cache_found:
        status = STATUS_OPTIONAL
        note = "HF token not set, but local model cache directory exists."
    else:
        status = STATUS_OPTIONAL
        note = (
            "HF_TOKEN / HUGGINGFACE_HUB_TOKEN not configured. "
            "Required only for downloading/running gated CohereLabs/tiny-aya-global; "
            "deterministic diagnosis remains available offline."
        )

    return {
        "component": "huggingface_access",
        "status": status,
        "huggingface_hub_installed": hf_pkg["installed"],
        "huggingface_hub_version": hf_pkg["version"],
        "token_present": token_present,
        "token_env_var": token_var,
        "local_model_cache_found": local_cache_found,
        "note": note,
    }


def check_gpu_availability() -> dict[str, Any]:
    """Check CUDA GPU presence, VRAM capacity, and system memory via psutil."""
    cuda_available = False
    gpu_name: str | None = None
    gpu_vram_gb = 0.0
    torch_info = _package_info("torch")

    if torch_info["installed"]:
        try:
            import torch

            cuda_available = bool(torch.cuda.is_available())
            if cuda_available:
                props = torch.cuda.get_device_properties(0)
                gpu_name = str(props.name)
                gpu_vram_gb = round(props.total_memory / (1024**3), 2)
        except Exception:
            cuda_available = False

    total_ram_gb = 0.0
    available_ram_gb = 0.0
    try:
        import psutil

        vm = psutil.virtual_memory()
        total_ram_gb = round(vm.total / (1024**3), 2)
        available_ram_gb = round(vm.available / (1024**3), 2)
    except Exception:
        pass

    if cuda_available and gpu_vram_gb >= 8.0:
        status = STATUS_READY
        note = f"CUDA GPU available ({gpu_name}, {gpu_vram_gb:.2f} GB VRAM)."
    elif cuda_available:
        status = STATUS_OPTIONAL
        note = (
            f"CUDA GPU detected ({gpu_name}, {gpu_vram_gb:.2f} GB VRAM), "
            "which may be tight for full fine-tuning/inference."
        )
    else:
        status = STATUS_OPTIONAL
        note = (
            "No CUDA GPU detected (CPU-only host). Live Tiny Aya training/inference is "
            "blocked without GPU/high RAM, while deterministic diagnostics and offline "
            "evaluation run normally."
        )

    return {
        "component": "gpu_and_memory",
        "status": status,
        "torch_installed": torch_info["installed"],
        "torch_version": torch_info["version"],
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "gpu_vram_gb": gpu_vram_gb,
        "total_ram_gb": total_ram_gb,
        "available_ram_gb": available_ram_gb,
        "note": note,
    }


def check_tiny_aya_availability(
    *,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Assess whether live CohereLabs/tiny-aya-global inference and fine-tuned weights exist."""
    hf_info = check_huggingface_access(env=env)
    gpu_info = check_gpu_availability()
    transformers_info = _package_info("transformers")
    accelerate_info = _package_info("accelerate")

    sft_dir = PROJECT_ROOT / "training" / "experiments" / "sft"
    checkpoint_exists = any(
        (sft_dir / fname).exists()
        for fname in ("adapter_model.safetensors", "adapter_model.bin", "model.safetensors")
    )

    blockers: list[str] = []
    if not transformers_info["installed"]:
        blockers.append("transformers package is not installed.")
    if not gpu_info["torch_installed"]:
        blockers.append("PyTorch is not installed.")
    if not hf_info["token_present"] and not hf_info["local_model_cache_found"]:
        blockers.append(
            "HF_TOKEN / HUGGINGFACE_HUB_TOKEN is not set for gated model CohereLabs/tiny-aya-global."
        )
    if not gpu_info["cuda_available"] and gpu_info["available_ram_gb"] < 8.0:
        blockers.append(
            f"Insufficient hardware for Tiny Aya CPU inference "
            f"(cuda=False, free_ram={gpu_info['available_ram_gb']:.2f} GB < 8.0 GB)."
        )

    live_ready = len(blockers) == 0
    return {
        "component": "tiny_aya_global",
        "status": STATUS_READY if live_ready else STATUS_OPTIONAL,
        "model_id": DEFAULT_MODEL_ID,
        "live_inference_ready": live_ready,
        "deterministic_fallback_supported": True,
        "fine_tuned_checkpoint_exists": checkpoint_exists,
        "fine_tuned_checkpoint_dir": str(sft_dir),
        "transformers_version": transformers_info["version"],
        "accelerate_version": accelerate_info["version"],
        "blockers": blockers,
        "note": (
            "Tiny Aya live inference prerequisites met."
            if live_ready
            else "Live Tiny Aya inference unavailable; deterministic DBMedic mode remains supported (explicit selection required, no silent fallback)."
        ),
    }


def check_agenttune_availability() -> dict[str, Any]:
    """Check AgentTune package version, TRL/PEFT stack, and curated training dataset."""
    at_info = _package_info("agenttune")
    trl_info = _package_info("trl")
    peft_info = _package_info("peft")
    manifest_path = PROJECT_ROOT / "training" / "data" / "dataset_manifest.json"

    status = STATUS_READY if at_info["installed"] else STATUS_OPTIONAL
    return {
        "component": "agenttune",
        "status": status,
        "installed": at_info["installed"],
        "version": at_info["version"],
        "trl_installed": trl_info["installed"],
        "trl_version": trl_info["version"],
        "peft_installed": peft_info["installed"],
        "peft_version": peft_info["version"],
        "dataset_manifest_exists": manifest_path.exists(),
        "note": (
            f"AgentTune v{at_info['version']} installed."
            if at_info["installed"]
            else "agenttune package is not installed; Phase 4 training adapters unavailable."
        ),
    }


def check_auditkit_availability() -> dict[str, Any]:
    """Distinguish real installed AuditKIT from DBMedic offline AuditKITStub."""
    pkg_info = _package_info("auditkit")
    adapter_available = False
    adapter_version = "not-installed"
    try:
        from agent.audit.auditkit_adapter import AUDITKIT_AVAILABLE, AUDITKIT_VERSION

        adapter_available = bool(AUDITKIT_AVAILABLE)
        adapter_version = str(AUDITKIT_VERSION)
    except Exception:
        pass

    is_stub = not adapter_available
    return {
        "component": "auditkit",
        "status": STATUS_READY if adapter_available else STATUS_OPTIONAL,
        "installed": bool(pkg_info["installed"] and adapter_available),
        "version": adapter_version,
        "using_stub": is_stub,
        "stub_name": "AuditKITStub" if is_stub else None,
        "license": "LSAL-1.2 (Lexsi Labs source-available, non-commercial; review before commercial deployment)",
        "note": (
            f"Real AuditKIT package active (v{adapter_version})."
            if not is_stub
            else "Real auditkit package is not installed; offline AuditKITStub active (NOT independent AuditKIT validation)."
        ),
    }


def generate_readiness_report(
    *,
    output_path: Path | str | None = None,
    timeout_sec: int = 2,
    env: dict[str, str] | None = None,
    conn_factory: Any = None,
    save: bool = True,
) -> dict[str, Any]:
    """
    Execute all DBMedic Phase 6 environment checks and optionally persist
    ``reports/environment_readiness.json``.
    """
    py_env = check_python_environment(env=env)
    db_conn = check_database_connection(
        timeout_sec=timeout_sec,
        env=env,
        conn_factory=conn_factory,
    )
    lab_schema = check_lab_schema(
        timeout_sec=timeout_sec,
        env=env,
        conn_factory=conn_factory,
    )
    hf_access = check_huggingface_access(env=env)
    gpu_info = check_gpu_availability()
    tiny_aya = check_tiny_aya_availability(env=env)
    agenttune = check_agenttune_availability()
    auditkit = check_auditkit_availability()

    live_db_ready = db_conn["status"] == STATUS_READY and lab_schema["status"] == STATUS_READY
    live_model_ready = bool(tiny_aya["live_inference_ready"])
    py_ready = py_env["status"] == STATUS_READY

    offline_features = [
        "unit_tests_and_safety_validators",
        "recorded_mode_auditkit_stub_evaluation",
        "deterministic_redteam_benchmark",
        "agenttune_curated_dataset_validation",
        "simulated_demo_workflow",
        "audit_trail_logging_and_verification",
    ]
    live_features = []
    if live_db_ready:
        live_features.extend(
            [
                "deterministic_live_postgres_diagnosis",
                "human_approved_index_remediation",
                "before_after_postgres_benchmark_verification",
            ]
        )
    if live_db_ready and live_model_ready:
        live_features.extend(["live_tiny_aya_agent_diagnosis", "live_tiny_aya_evaluation"])

    if py_ready and live_db_ready and live_model_ready:
        overall_status = STATUS_READY
    elif py_ready:
        overall_status = "PARTIAL_READY"
    else:
        overall_status = STATUS_BLOCKED

    report: dict[str, Any] = {
        "schema_version": "dbmedic.environment_readiness/1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "overall_status": overall_status,
        "capabilities": {
            "python_core_ready": py_ready,
            "live_postgres_ready": live_db_ready,
            "deterministic_db_mode_ready": py_ready and live_db_ready,
            "live_tiny_aya_ready": py_ready and live_model_ready,
            "full_live_ai_workflow_ready": py_ready and live_db_ready and live_model_ready,
            "offline_mode_ready": py_ready,
            "auditkit_real_installed": auditkit["installed"],
            "agenttune_installed": agenttune["installed"],
            "streamlit_installed": py_env["optional_packages"]["streamlit"]["installed"],
        },
        "offline_features_available": offline_features,
        "live_features_available": live_features,
        "checks": {
            "python_environment": py_env,
            "database_connection": db_conn,
            "lab_schema": lab_schema,
            "huggingface_access": hf_access,
            "tiny_aya_global": tiny_aya,
            "agenttune": agenttune,
            "auditkit": auditkit,
            "gpu_and_memory": gpu_info,
        },
        "blockers": [
            b
            for b in [
                *py_env.get("blockers", []),
                db_conn.get("blocker"),
                lab_schema.get("blocker") if db_conn["connected"] else None,
                *tiny_aya.get("blockers", []),
            ]
            if b
        ],
    }

    report = redact_secrets(report)

    if save:
        dest = (
            Path(output_path)
            if output_path is not None
            else DEFAULT_READINESS_REPORT_PATH
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        report["report_path"] = str(dest)

    return report
