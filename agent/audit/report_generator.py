"""
agent/audit/report_generator.py
=================================
Generates structured AuditKIT evaluation reports in JSON and Markdown.

Output:
  benchmarks/auditkit/reports/final_evaluation_report.json
  benchmarks/auditkit/reports/final_evaluation_report.md

Report includes:
  - AuditKIT version and DBMedic version/commit
  - Base and candidate model identifiers
  - Evaluation dataset hashes
  - Sample counts
  - Metric definitions and results
  - Red-team results
  - Performance results
  - Regression analysis
  - Deployment gate decision
  - Unresolved blockers

Synthetic fixtures, mocked tests, and pending evaluations are
explicitly labeled and never conflated with actual model results.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _get_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _hash_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


class ReportGenerator:
    """
    Generates final AuditKIT evaluation reports in JSON and Markdown.
    """

    DBMEDIC_VERSION = "0.5.0-phase5"

    # Metric definitions (for report header)
    METRIC_DEFINITIONS = {
        "tool_call_validity_rate": "Fraction of model tool calls that passed the validator.",
        "tool_selection_accuracy": "Fraction of runs selecting at least one relevant diagnostic tool.",
        "task_success_rate": "Fraction of runs producing an independently verified correct diagnosis.",
        "average_tool_calls_per_task": "Mean number of tool invocations per evaluation episode.",
        "average_steps_per_task": "Mean total agent steps (model + tool) per episode.",
        "safety_violation_count": "Total number of unsafe/destructive tool calls rejected.",
        "evidence_grounding_accuracy": "Fraction of evidence items grounded in actual tool observations.",
        "diagnosis_accuracy": "Exact root-cause match against independently verified ground truth.",
        "redteam_attack_success_rate": "Fraction of red-team attacks that succeeded past all defenses.",
        "inference_latency_ms": "Mean model inference latency in milliseconds.",
        "end_to_end_latency_ms": "Mean end-to-end agent latency in milliseconds.",
        "tool_execution_failures": "Number of tool executions that produced an error result.",
        "evaluation_sample_count": "Number of unique evaluation episodes processed.",
    }

    def generate(
        self,
        *,
        auditkit_version: str,
        base_model_id: str,
        base_model_revision: str | None,
        candidate_model_id: str | None,
        candidate_model_revision: str | None,
        baseline_results: dict[str, Any] | None,
        candidate_results: dict[str, Any] | None,
        redteam_results: dict[str, Any] | None,
        performance_results: dict[str, Any] | None,
        regression_report: dict[str, Any] | None,
        deployment_gate: dict[str, Any] | None,
        evaluation_dataset_paths: list[Path] | None = None,
        blockers: list[str] | None = None,
        output_dir: Path | None = None,
    ) -> dict[str, str]:
        """
        Generate and write the final evaluation reports.

        Returns
        -------
        dict with 'json_path' and 'md_path' keys.
        """
        output_dir = output_dir or (PROJECT_ROOT / "benchmarks" / "auditkit" / "reports")
        output_dir.mkdir(parents=True, exist_ok=True)

        commit = _get_git_commit()
        timestamp = datetime.now(timezone.utc).isoformat()

        # Dataset hashes
        dataset_hashes: dict[str, str | None] = {}
        for dp in (evaluation_dataset_paths or []):
            dataset_hashes[str(dp)] = _hash_file(Path(dp))

        report = {
            "schema": "dbmedic_auditkit_report/1",
            "generated_at": timestamp,
            "dbmedic_version": self.DBMEDIC_VERSION,
            "dbmedic_commit": commit,
            "auditkit_version": auditkit_version,
            "base_model": {
                "id": base_model_id,
                "revision": base_model_revision,
            },
            "candidate_model": {
                "id": candidate_model_id,
                "revision": candidate_model_revision,
            },
            "evaluation_dataset": {
                "hashes": dataset_hashes,
                "note": "Ground-truth answers are never exposed to the evaluated model.",
            },
            "evaluation_sample_counts": self._extract_sample_counts(
                baseline_results, candidate_results, redteam_results
            ),
            "metric_definitions": self.METRIC_DEFINITIONS,
            "baseline_results": self._label_results(
                baseline_results, "base_model_evaluation"
            ),
            "candidate_results": self._label_results(
                candidate_results, "candidate_model_evaluation"
            ),
            "redteam_results": self._label_results(
                redteam_results, "red_team_evaluation"
            ),
            "performance_results": self._label_results(
                performance_results, "performance_evaluation"
            ),
            "regression_analysis": regression_report or {
                "status": "pending",
                "reason": "One or both model evaluations have not completed.",
            },
            "deployment_gate": deployment_gate or {
                "decision": "hold_keep_base",
                "reason": "Real evaluation data not yet available.",
                "approved": False,
            },
            "unresolved_blockers": blockers or [],
        }

        json_path = output_dir / "final_evaluation_report.json"
        md_path = output_dir / "final_evaluation_report.md"

        json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        logger.info("Wrote JSON report to %s", json_path)

        md_content = self._render_markdown(report)
        md_path.write_text(md_content, encoding="utf-8")
        logger.info("Wrote Markdown report to %s", md_path)

        return {"json_path": str(json_path), "md_path": str(md_path)}

    # ------------------------------------------------------------------
    # Markdown rendering
    # ------------------------------------------------------------------

    def _render_markdown(self, report: dict[str, Any]) -> str:
        lines = [
            "# DBMedic — Phase 5 AuditKIT Evaluation Report",
            "",
            f"**Generated:** {report['generated_at']}  ",
            f"**DBMedic Version:** {report['dbmedic_version']} (commit: `{report['dbmedic_commit']}`)  ",
            f"**AuditKIT Version:** {report['auditkit_version']}  ",
            "",
            "---",
            "",
            "## 1. Model Identifiers",
            "",
            f"| Role | Model ID | Revision |",
            f"|------|----------|----------|",
            f"| Base | `{report['base_model']['id']}` | `{report['base_model']['revision'] or 'N/A'}` |",
            f"| Candidate | `{report['candidate_model']['id'] or 'N/A'}` | `{report['candidate_model']['revision'] or 'N/A'}` |",
            "",
            "---",
            "",
            "## 2. Evaluation Dataset",
            "",
        ]

        hashes = report["evaluation_dataset"]["hashes"]
        if hashes:
            lines += ["| Dataset File | SHA-256 |", "|---|---|"]
            for path, h in hashes.items():
                lines.append(f"| `{Path(path).name}` | `{h or 'N/A'}` |")
        else:
            lines.append("*No dataset files recorded.*")

        lines += [
            "",
            "---",
            "",
            "## 3. Evaluation Sample Counts",
            "",
        ]
        counts = report["evaluation_sample_counts"]
        for k, v in counts.items():
            lines.append(f"- **{k}:** {v}")

        lines += ["", "---", "", "## 4. Metric Definitions", ""]
        for k, v in report["metric_definitions"].items():
            lines.append(f"- **`{k}`**: {v}")

        lines += ["", "---", "", "## 5. Baseline (Base Model) Results", ""]
        lines += self._render_results_section(report["baseline_results"])

        lines += ["", "---", "", "## 6. Candidate Model Results", ""]
        lines += self._render_results_section(report["candidate_results"])

        lines += ["", "---", "", "## 7. Red-Team Evaluation Results", ""]
        lines += self._render_results_section(report["redteam_results"])

        lines += ["", "---", "", "## 8. Performance Results", ""]
        lines += self._render_results_section(report["performance_results"])

        lines += ["", "---", "", "## 9. Regression Analysis", ""]
        reg = report["regression_analysis"]
        if isinstance(reg, dict) and "overall_verdict" in reg:
            verdict = reg.get("overall_verdict", "unknown")
            lines.append(f"**Verdict:** `{verdict}`")
            regressions = reg.get("regressions", [])
            if regressions:
                lines += ["", "### Detected Regressions", ""]
                for r in regressions:
                    lines.append(
                        f"- ⚠️ **{r['metric']}**: {r['baseline']} → {r['candidate']} "
                        f"(Δ={r['delta']:+.4f}, severity={r.get('severity','?')})"
                    )
            else:
                lines.append("No regressions detected.")
        else:
            lines.append(f"**Status:** {reg.get('status', 'unknown')} — {reg.get('reason', '')}")

        lines += ["", "---", "", "## 10. Deployment Gate Decision", ""]
        gate = report["deployment_gate"]
        decision = gate.get("decision", "unknown")
        approved = gate.get("approved", False)
        lines += [
            f"**Decision:** `{decision}`  ",
            f"**Approved:** `{approved}`  ",
            f"**Reason:** {gate.get('reason', 'N/A')}",
            "",
            "> [!IMPORTANT]",
            "> Automatic deployment is never permitted. Human approval is mandatory.",
        ]

        lines += ["", "---", "", "## 11. Unresolved Blockers", ""]
        blockers = report["unresolved_blockers"]
        if blockers:
            for b in blockers:
                lines.append(f"- ❌ {b}")
        else:
            lines.append("*No unresolved blockers.*")

        return "\n".join(lines) + "\n"

    def _render_results_section(self, results: dict[str, Any] | None) -> list[str]:
        if results is None:
            return ["*Not evaluated yet — no data available.*"]
        status = results.get("_status", "actual")
        if status in {"pending", "not_available"}:
            return [f"*{results.get('_reason', 'No data available.')}*"]

        lines = []
        for k, v in results.items():
            if k.startswith("_"):
                continue
            if isinstance(v, dict) and "value" in v:
                val = v["value"]
                scored = v.get("scored", True)
                if not scored:
                    lines.append(f"- **{k}:** `null` _(reason: {v.get('reason', 'unscored')})_")
                else:
                    lines.append(f"- **{k}:** `{val}`")
            else:
                lines.append(f"- **{k}:** `{v}`")
        return lines or ["*No metrics recorded.*"]

    @staticmethod
    def _label_results(
        results: dict[str, Any] | None,
        label: str,
    ) -> dict[str, Any]:
        if results is None:
            return {
                "_status": "not_available",
                "_label": label,
                "_reason": f"{label} has not been completed. All metrics are pending.",
            }
        return {**results, "_label": label, "_status": "actual"}

    @staticmethod
    def _extract_sample_counts(
        baseline: dict[str, Any] | None,
        candidate: dict[str, Any] | None,
        redteam: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return {
            "baseline_episodes": (baseline or {}).get("num_episodes", "pending"),
            "candidate_episodes": (candidate or {}).get("num_episodes", "pending"),
            "redteam_cases": (redteam or {}).get("num_cases", "pending"),
        }

