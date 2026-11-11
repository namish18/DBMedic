"""
agent/audit/performance_metrics.py
====================================
Performance measurement for DBMedic agent evaluation.

Measures:
  - Model inference latency (separate from PostgreSQL execution latency)
  - End-to-end agent latency
  - Average response time
  - Tool execution time
  - Number of model calls
  - Number of tool calls
  - Throughput (where measurable)
  - Model and adapter size (where available)

Hardware and generation configuration are recorded for reproducibility.
Latency results from different hardware are NEVER compared as equivalent.
"""

from __future__ import annotations

import logging
import statistics
from typing import Any

logger = logging.getLogger(__name__)


def compute_performance_metrics(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Compute performance metrics from episode dicts.

    Parameters
    ----------
    episodes : list[dict]
        Episode.to_dict() records.

    Returns
    -------
    dict mapping metric name to scored value dict.
    """
    if not episodes:
        return _empty_perf_metrics()

    total_e2e_ms: list[float] = []
    model_inference_ms: list[float] = []
    tool_exec_ms: list[float] = []
    model_call_counts: list[int] = []
    tool_call_counts: list[int] = []
    step_gen_times: list[float] = []

    hardware_records: list[dict[str, Any]] = []
    generation_configs: list[dict[str, Any]] = []

    for ep in episodes:
        e2e = ep.get("total_time_ms")
        if e2e is not None and isinstance(e2e, (int, float)):
            total_e2e_ms.append(float(e2e))

        model_ms = ep.get("model_inference_time_ms")
        if model_ms is not None and isinstance(model_ms, (int, float)):
            model_inference_ms.append(float(model_ms))

        tool_ms = ep.get("tool_execution_time_ms")
        if tool_ms is not None and isinstance(tool_ms, (int, float)):
            tool_exec_ms.append(float(tool_ms))

        n_model = ep.get("num_model_calls", 0)
        if isinstance(n_model, int):
            model_call_counts.append(n_model)

        n_tool = ep.get("num_tool_calls", 0)
        if isinstance(n_tool, int):
            tool_call_counts.append(n_tool)

        # Per-step generation times
        for step in ep.get("steps", []):
            gen_ms = step.get("generation_time_ms")
            if gen_ms is not None and isinstance(gen_ms, (int, float)):
                step_gen_times.append(float(gen_ms))

        hw = ep.get("hardware_info") or {}
        if hw:
            hardware_records.append(hw)

        gc = ep.get("generation_config") or {}
        if gc:
            generation_configs.append(gc)

    def _stat(values: list[float], label: str) -> dict[str, Any]:
        if not values:
            return {
                "value": None,
                "reason": f"no_{label}_data_recorded",
                "scored": False,
            }
        return {
            "value": round(statistics.mean(values), 2),
            "min_ms": round(min(values), 2),
            "max_ms": round(max(values), 2),
            "median_ms": round(statistics.median(values), 2),
            "stddev_ms": round(statistics.stdev(values), 2) if len(values) > 1 else 0.0,
            "num_samples": len(values),
            "scored": True,
        }

    def _avg_int(values: list[int]) -> dict[str, Any]:
        if not values:
            return {"value": None, "reason": "no_data", "scored": False}
        return {
            "value": round(sum(values) / len(values), 2),
            "total": sum(values),
            "scored": True,
        }

    # Throughput: episodes / total wall time (approximate)
    throughput: dict[str, Any]
    if total_e2e_ms:
        total_wall_sec = sum(total_e2e_ms) / 1000.0
        throughput = {
            "value": round(len(episodes) / total_wall_sec, 4) if total_wall_sec > 0 else None,
            "unit": "episodes_per_second",
            "num_episodes": len(episodes),
            "total_wall_sec": round(total_wall_sec, 2),
            "scored": total_wall_sec > 0,
        }
    else:
        throughput = {"value": None, "reason": "no_latency_data", "scored": False}

    # Hardware and generation config summary
    hw_summary = _summarize_hardware(hardware_records)
    gen_summary = _summarize_generation_configs(generation_configs)

    result = {
        "latency_stats": {
            "end_to_end_ms": _stat(total_e2e_ms, "e2e_latency"),
            "model_inference_ms": _stat(model_inference_ms, "model_inference"),
            "tool_execution_ms": _stat(tool_exec_ms, "tool_execution"),
            "per_step_generation_ms": _stat(step_gen_times, "step_generation"),
            "note": (
                "Model inference latency and PostgreSQL execution latency are "
                "measured separately. Do not compare values from different hardware."
            ),
            "hardware": hw_summary,
            "generation_config": gen_summary,
            "scored": True,
        },
        "model_call_counts": _avg_int(model_call_counts),
        "tool_call_counts": _avg_int(tool_call_counts),
        "throughput": throughput,
    }

    return result


def _empty_perf_metrics() -> dict[str, Any]:
    return {
        "latency_stats": {
            "value": None,
            "reason": "no_episodes",
            "scored": False,
        },
        "model_call_counts": {"value": None, "reason": "no_episodes", "scored": False},
        "tool_call_counts": {"value": None, "reason": "no_episodes", "scored": False},
        "throughput": {"value": None, "reason": "no_episodes", "scored": False},
    }


def _summarize_hardware(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"available": False, "reason": "hardware_info_not_recorded"}
    # Use first record as representative
    first = records[0]
    return {
        "available": True,
        "device": first.get("device", "unknown"),
        "cuda_available": first.get("cuda_available", False),
        "cpu_count": first.get("cpu_count"),
        "ram_gb": first.get("ram_gb"),
        "gpu_name": first.get("gpu_name"),
        "gpu_vram_gb": first.get("gpu_vram_gb"),
        "note": "Results are only comparable on identical hardware configurations.",
    }


def _summarize_generation_configs(configs: list[dict[str, Any]]) -> dict[str, Any]:
    if not configs:
        return {"available": False, "reason": "generation_config_not_recorded"}
    first = configs[0]
    return {
        "available": True,
        "do_sample": first.get("do_sample", False),
        "temperature": first.get("temperature"),
        "max_new_tokens": first.get("max_new_tokens"),
        "top_p": first.get("top_p"),
    }


def collect_hardware_info() -> dict[str, Any]:
    """
    Collect current hardware configuration for performance records.
    Safe to call without CUDA or GPU.
    """
    import os
    info: dict[str, Any] = {}
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / (1024**3), 2)
        info["cpu_count"] = psutil.cpu_count(logical=True)
    except ImportError:
        info["ram_gb"] = None
        info["cpu_count"] = os.cpu_count()

    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        if info["cuda_available"]:
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_vram_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / (1024**3), 2
            )
            info["device"] = "cuda"
        else:
            info["device"] = "cpu"
            info["gpu_name"] = None
            info["gpu_vram_gb"] = None
    except ImportError:
        info["cuda_available"] = False
        info["device"] = "cpu"

    return info

