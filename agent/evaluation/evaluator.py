"""
agent/evaluation/evaluator.py
==============================
Runs baseline evaluation of the DBMedic agent on reproducible incident cases.

Creates:
  benchmarks/baseline/results.json      — per-case structured results
  benchmarks/baseline/trajectories.jsonl — JSONL trajectory log
  benchmarks/baseline/summary.json      — aggregate metrics summary

No fabricated numbers — all results derive from actual model inference.
Evaluation is deterministic (do_sample=False) for reproducibility.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BASELINE_DIR = PROJECT_ROOT / "benchmarks" / "baseline"


class BaselineEvaluator:
    """
    Runs baseline evaluation of the DBMedic agent on the incident dataset.
    """

    def __init__(
        self,
        agent: Any,
        output_dir: Path | str = BASELINE_DIR,
        runs_per_case: int = 1,
    ) -> None:
        self.agent = agent
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.runs_per_case = runs_per_case

    # ------------------------------------------------------------------
    # Main evaluation entry point
    # ------------------------------------------------------------------

    def run_baseline(
        self,
        cases: list[Any] | None = None,
        max_cases: int | None = None,
    ) -> dict[str, Any]:
        """
        Run baseline evaluation on all or a subset of incident cases.

        Parameters
        ----------
        cases: list[IncidentCase] | None
            Cases to evaluate. Defaults to BASELINE_EVAL_CASES.
        max_cases: int | None
            Maximum number of cases to run (for quick testing).

        Returns
        -------
        dict: Evaluation summary with all metrics.
        """
        from agent.evaluation.incident_cases import BASELINE_EVAL_CASES
        from agent.evaluation.metrics import compute_all_metrics, format_metrics_report
        from agent.evaluation.trajectory_logger import TrajectoryLogger

        if cases is None:
            cases = BASELINE_EVAL_CASES
        if max_cases is not None:
            cases = cases[:max_cases]

        logger.info(
            "Starting baseline evaluation: %d cases x %d run(s) = %d total runs",
            len(cases), self.runs_per_case, len(cases) * self.runs_per_case,
        )

        trajectory_logger = TrajectoryLogger(self.output_dir)
        all_results: list[dict[str, Any]] = []
        per_case_results: list[dict[str, Any]] = []

        eval_start = time.perf_counter()

        for case_idx, case in enumerate(cases):
            logger.info(
                "Evaluating case %d/%d: %s",
                case_idx + 1, len(cases), case.incident_id,
            )

            case_runs: list[dict[str, Any]] = []

            for run_idx in range(self.runs_per_case):
                logger.info(
                    "  Run %d/%d for %s",
                    run_idx + 1, self.runs_per_case, case.incident_id,
                )
                try:
                    result = self.agent.run(
                        query=case.query,
                        incident_id=f"{case.incident_id}_run{run_idx + 1}",
                        log_trajectory=False,  # We log manually below
                    )
                    result_dict = result.to_dict()

                    # Add trajectory steps for evidence grounding (not in to_dict)
                    result_dict["_trajectory_steps"] = [
                        {
                            "step_type": s.step_type,
                            "tool_name": s.tool_name,
                            "tool_output": s.tool_output,
                        }
                        for s in result.trajectory
                    ]

                    # Add ground truth comparison
                    result_dict["ground_truth"] = {
                        "expected_root_cause": case.expected_root_cause,
                        "expected_tools": case.expected_tools_to_call,
                        "diagnosis_correct": (
                            result_dict.get("root_cause", "").lower()
                            == case.expected_root_cause.lower()
                        ),
                        "used_relevant_tool": bool(
                            set(result_dict.get("tools_called", []))
                            & set(case.expected_tools_to_call)
                        ),
                    }

                    case_runs.append(result_dict)
                    all_results.append(result_dict)

                    # Log trajectory
                    trajectory_logger.log(result)

                except Exception as exc:
                    logger.error(
                        "Case %s run %d failed: %s",
                        case.incident_id, run_idx + 1, exc,
                        exc_info=True,
                    )
                    # Record failure
                    failure = {
                        "incident_id": f"{case.incident_id}_run{run_idx + 1}",
                        "query": case.query,
                        "success": False,
                        "diagnosis": "",
                        "root_cause": "unknown",
                        "evidence": [],
                        "tools_called": [],
                        "num_steps": 0,
                        "safety_violations": [],
                        "rejection_reasons": [],
                        "error": str(exc),
                        "total_time_ms": 0,
                        "metadata": {},
                        "ground_truth": {
                            "expected_root_cause": case.expected_root_cause,
                            "diagnosis_correct": False,
                        },
                        "_trajectory_steps": [],
                    }
                    case_runs.append(failure)
                    all_results.append(failure)

            # Aggregate case-level results
            per_case_results.append({
                "incident_id": case.incident_id,
                "query": case.query,
                "split": case.split,
                "num_runs": len(case_runs),
                "runs": case_runs,
                "case_success_rate": sum(
                    1 for r in case_runs if r.get("success", False)
                ) / max(1, len(case_runs)),
                "ground_truth": {
                    "expected_root_cause": case.expected_root_cause,
                    "expected_tools": case.expected_tools_to_call,
                },
            })

        total_eval_time_ms = (time.perf_counter() - eval_start) * 1000

        # Compute aggregate metrics
        metrics = compute_all_metrics(
            all_results,
            expected_root_cause="missing_suitable_index",
        )

        # Build summary
        summary = {
            "evaluation_id": f"baseline_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_id": getattr(self.agent, "model_id", "unknown"),
            "model_revision": getattr(
                getattr(self.agent, "_model_client", None), "model_revision", None
            ),
            "total_cases": len(cases),
            "runs_per_case": self.runs_per_case,
            "total_runs": len(all_results),
            "total_eval_time_ms": round(total_eval_time_ms, 1),
            "metrics": metrics,
            "generation_config": getattr(self.agent, "generation_config", {}),
            "note": (
                "Baseline evaluation before AgentTune training. "
                "do_sample=False for deterministic reproducibility."
            ),
        }

        # Write outputs
        self._write_results(per_case_results)
        self._write_summary(summary)

        # Print report
        report = format_metrics_report(metrics)
        print(report)
        logger.info("Baseline evaluation complete. Results saved to %s", self.output_dir)

        return summary

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    def _write_results(self, per_case_results: list[dict[str, Any]]) -> None:
        results_path = self.output_dir / "results.json"
        # Remove internal _trajectory_steps from JSON output
        clean = json.loads(json.dumps(per_case_results, default=str))
        for case in clean:
            for run in case.get("runs", []):
                run.pop("_trajectory_steps", None)

        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(clean, f, indent=2, ensure_ascii=False)
        logger.info("Per-case results written to %s", results_path)

    def _write_summary(self, summary: dict[str, Any]) -> None:
        summary_path = self.output_dir / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info("Evaluation summary written to %s", summary_path)

