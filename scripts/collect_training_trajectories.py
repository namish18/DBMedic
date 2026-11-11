"""
scripts/collect_training_trajectories.py
========================================
Collect DBMedic agent trajectories into `training/data/raw/` for failure analysis
and AgentTune dataset curation.

Modes:
  1. Collect existing real trajectories from `benchmarks/baseline/trajectories.jsonl`
     or custom `--source` JSONL files.
  2. `--live-run`: execute `DBMedicAgent` on `TRAINING_CANDIDATES` (requires
     HF_TOKEN and reachable `dbmedic_lab`).
  3. `--generate-controlled-failures-sample`: create a controlled synthetic failure
     sample covering all 10 failure categories in
     `training/data/raw/sample_controlled_trajectories.jsonl`, explicitly tagged
     `simulated=True` (never modifies `benchmarks/baseline/`).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent.evaluation.incident_cases import TRAINING_CANDIDATES
from training.agenttune.distillation import _detect_hf_token, _probe_db_available
from training.failure_analysis.analyzer import load_trajectories_jsonl


def _build_controlled_sample_trajectories() -> list[dict[str, Any]]:
    """Build controlled synthetic sample covering all 10 DBMedic failure categories + successes."""
    now = datetime.now(timezone.utc).isoformat()
    base_plan_out = {
        "metrics": {
            "scan_type": "Sequential Scan",
            "execution_time_ms": 18.42,
            "actual_rows": 22,
            "shared_hit_blocks": 1250,
        },
        "scan_analysis": {"primary_scan_type": "Seq Scan"},
    }
    base_idx_out = {
        "table_name": "orders",
        "target_column": "customer_id",
        "index_exists": False,
        "is_suitable_for_filter": False,
    }

    samples: list[dict[str, Any]] = [
        # 1. malformed_tool_call
        {
            "trajectory_id": "sample_fail_01",
            "incident_id": "train_001",
            "query": TRAINING_CANDIDATES[0].query,
            "simulated": True,
            "source": "controlled_failure_simulation",
            "timestamp": now,
            "tools_called": [],
            "num_steps": 2,
            "success": False,
            "diagnosis": "",
            "root_cause": "unknown",
            "evidence": [],
            "recommended_action": "",
            "requires_approval": True,
            "safety_violations": [],
            "trajectory_steps": [
                {
                    "step_index": 1,
                    "step_type": "model_response",
                    "model_response": "I will run SELECT * FROM orders WHERE customer_id = 7777;",
                    "is_valid": False,
                    "validation_error": "Model did not return valid JSON object",
                }
            ],
        },
        # 2. invalid_tool_arguments
        {
            "trajectory_id": "sample_fail_02",
            "incident_id": "train_002",
            "query": TRAINING_CANDIDATES[1].query,
            "simulated": True,
            "source": "controlled_failure_simulation",
            "timestamp": now,
            "tools_called": [],
            "num_steps": 2,
            "success": False,
            "diagnosis": "",
            "root_cause": "unknown",
            "evidence": [],
            "recommended_action": "",
            "requires_approval": True,
            "safety_violations": [],
            "trajectory_steps": [
                {
                    "step_index": 1,
                    "step_type": "model_response",
                    "model_response": json.dumps(
                        {
                            "type": "tool_call",
                            "tool": "get_table_statistics",
                            "arguments": {"table_name": "users_secret_table"},
                            "reasoning": "Inspect table",
                        }
                    ),
                    "tool_name": "get_table_statistics",
                    "tool_arguments": {"table_name": "users_secret_table"},
                    "is_valid": False,
                    "validation_error": "Table 'users_secret_table' is not in ALLOWED_TABLE_NAMES",
                }
            ],
        },
        # 3. incorrect_tool_selection
        {
            "trajectory_id": "sample_fail_03",
            "incident_id": "train_004",
            "query": TRAINING_CANDIDATES[3].query,
            "simulated": True,
            "source": "controlled_failure_simulation",
            "timestamp": now,
            "tools_called": ["get_table_size"],
            "num_steps": 3,
            "success": True,
            "diagnosis": "The orders table size was inspected.",
            "root_cause": "missing_suitable_index",
            "evidence": ["orders total size checked"],
            "recommended_action": "CREATE INDEX idx_orders_customer_id ON orders(customer_id)",
            "requires_approval": True,
            "trajectory_steps": [
                {
                    "step_index": 1,
                    "step_type": "tool_execution",
                    "model_response": json.dumps(
                        {"type": "tool_call", "tool": "get_table_size", "arguments": {"table_name": "orders"}}
                    ),
                    "tool_name": "get_table_size",
                    "tool_arguments": {"table_name": "orders"},
                    "tool_output": {"table_name": "orders", "total_size_pretty": "34 MB"},
                    "is_valid": True,
                },
                {
                    "step_index": 2,
                    "step_type": "final_answer",
                    "model_response": json.dumps(
                        {
                            "type": "final_answer",
                            "diagnosis": "The orders table size is 34 MB and customer_id lacks an index.",
                            "root_cause": "missing_suitable_index",
                            "evidence": ["orders total size is 34 MB"],
                            "recommended_action": "CREATE INDEX idx_orders_customer_id ON orders(customer_id)",
                            "requires_approval": True,
                        }
                    ),
                    "is_valid": True,
                },
            ],
        },
        # 4. unnecessary_tool_call
        {
            "trajectory_id": "sample_fail_04",
            "incident_id": "train_003",
            "query": TRAINING_CANDIDATES[2].query,
            "simulated": True,
            "source": "controlled_failure_simulation",
            "timestamp": now,
            "tools_called": [
                "analyze_query_performance",
                "analyze_query_performance",
                "inspect_customer_id_index",
            ],
            "num_steps": 4,
            "success": True,
            "diagnosis": "Sequential scan on orders due to missing index on customer_id.",
            "root_cause": "missing_suitable_index",
            "evidence": ["Seq Scan observed (18.42 ms)", "index_exists is false for customer_id"],
            "recommended_action": "CREATE INDEX idx_orders_customer_id ON orders(customer_id)",
            "requires_approval": True,
            "trajectory_steps": [
                {
                    "step_index": 1,
                    "step_type": "tool_execution",
                    "model_response": json.dumps({"type": "tool_call", "tool": "analyze_query_performance", "arguments": {"customer_id": 5150}}),
                    "tool_name": "analyze_query_performance",
                    "tool_arguments": {"customer_id": 5150},
                    "tool_output": base_plan_out,
                    "is_valid": True,
                },
                {
                    "step_index": 2,
                    "step_type": "tool_execution",
                    "model_response": json.dumps({"type": "tool_call", "tool": "analyze_query_performance", "arguments": {"customer_id": 5150}}),
                    "tool_name": "analyze_query_performance",
                    "tool_arguments": {"customer_id": 5150},
                    "tool_output": base_plan_out,
                    "is_valid": True,
                },
                {
                    "step_index": 3,
                    "step_type": "tool_execution",
                    "model_response": json.dumps({"type": "tool_call", "tool": "inspect_customer_id_index", "arguments": {}}),
                    "tool_name": "inspect_customer_id_index",
                    "tool_arguments": {},
                    "tool_output": base_idx_out,
                    "is_valid": True,
                },
                {
                    "step_index": 4,
                    "step_type": "final_answer",
                    "model_response": json.dumps(
                        {
                            "type": "final_answer",
                            "diagnosis": "Sequential scan on orders due to missing index on customer_id.",
                            "root_cause": "missing_suitable_index",
                            "evidence": ["Seq Scan in 18.42 ms", "orders.customer_id index_exists false"],
                            "recommended_action": "CREATE INDEX idx_orders_customer_id ON orders(customer_id)",
                            "requires_approval": True,
                        }
                    ),
                    "is_valid": True,
                },
            ],
        },
        # 5. incomplete_diagnosis
        {
            "trajectory_id": "sample_fail_05",
            "incident_id": "train_005",
            "query": TRAINING_CANDIDATES[4].query,
            "simulated": True,
            "source": "controlled_failure_simulation",
            "timestamp": now,
            "tools_called": ["analyze_query_performance", "get_table_statistics"],
            "num_steps": 3,
            "success": True,
            "diagnosis": "",
            "root_cause": "unknown",
            "evidence": [],
            "recommended_action": "",
            "requires_approval": True,
            "trajectory_steps": [
                {
                    "step_index": 1,
                    "step_type": "tool_execution",
                    "model_response": json.dumps({"type": "tool_call", "tool": "analyze_query_performance", "arguments": {"customer_id": 4242}}),
                    "tool_name": "analyze_query_performance",
                    "tool_arguments": {"customer_id": 4242},
                    "tool_output": base_plan_out,
                    "is_valid": True,
                },
                {
                    "step_index": 2,
                    "step_type": "final_answer",
                    "model_response": json.dumps({"type": "final_answer", "diagnosis": "", "root_cause": "unknown", "evidence": []}),
                    "is_valid": True,
                },
            ],
        },
        # 6. incorrect_root_cause
        {
            "trajectory_id": "sample_fail_06",
            "incident_id": "train_006",
            "query": TRAINING_CANDIDATES[5].query,
            "simulated": True,
            "source": "controlled_failure_simulation",
            "timestamp": now,
            "tools_called": ["get_index_usage", "analyze_query_performance"],
            "num_steps": 3,
            "success": True,
            "diagnosis": "The query suffers from network congestion.",
            "root_cause": "network_latency",
            "evidence": ["Seq Scan 18.42 ms on orders"],
            "recommended_action": "Upgrade NIC",
            "requires_approval": True,
            "trajectory_steps": [
                {
                    "step_index": 1,
                    "step_type": "tool_execution",
                    "model_response": json.dumps({"type": "tool_call", "tool": "analyze_query_performance", "arguments": {"customer_id": 4242}}),
                    "tool_name": "analyze_query_performance",
                    "tool_arguments": {"customer_id": 4242},
                    "tool_output": base_plan_out,
                    "is_valid": True,
                },
                {
                    "step_index": 2,
                    "step_type": "final_answer",
                    "model_response": json.dumps(
                        {
                            "type": "final_answer",
                            "diagnosis": "Network congestion caused slow orders query.",
                            "root_cause": "network_latency",
                            "evidence": ["Seq Scan 18.42 ms on orders"],
                            "recommended_action": "Upgrade NIC",
                            "requires_approval": True,
                        }
                    ),
                    "is_valid": True,
                },
            ],
        },
        # 7. unsupported_claim
        {
            "trajectory_id": "sample_fail_07",
            "incident_id": "train_007",
            "query": TRAINING_CANDIDATES[6].query,
            "simulated": True,
            "source": "controlled_failure_simulation",
            "timestamp": now,
            "tools_called": ["inspect_customer_id_index"],
            "num_steps": 2,
            "success": True,
            "diagnosis": "An index exists on customer_id and returned 1 row via Hash Join.",
            "root_cause": "missing_suitable_index",
            "evidence": ["An active index exists on orders(customer_id) using Hash Join."],
            "recommended_action": "CREATE INDEX idx_orders_customer_id ON orders(customer_id)",
            "requires_approval": True,
            "trajectory_steps": [
                {
                    "step_index": 1,
                    "step_type": "tool_execution",
                    "model_response": json.dumps({"type": "tool_call", "tool": "inspect_customer_id_index", "arguments": {}}),
                    "tool_name": "inspect_customer_id_index",
                    "tool_arguments": {},
                    "tool_output": base_idx_out,
                    "is_valid": True,
                },
                {
                    "step_index": 2,
                    "step_type": "final_answer",
                    "model_response": json.dumps(
                        {
                            "type": "final_answer",
                            "diagnosis": "An index exists on customer_id.",
                            "root_cause": "missing_suitable_index",
                            "evidence": ["Hash Join scanned cached heap partition 999GB."],
                            "recommended_action": "CREATE INDEX idx_orders_customer_id ON orders(customer_id)",
                            "requires_approval": True,
                        }
                    ),
                    "is_valid": True,
                },
            ],
        },
        # 8. excessive_agent_steps
        {
            "trajectory_id": "sample_fail_08",
            "incident_id": "train_008",
            "query": TRAINING_CANDIDATES[7].query,
            "simulated": True,
            "source": "controlled_failure_simulation",
            "timestamp": now,
            "tools_called": [
                "analyze_query_performance",
                "get_table_statistics",
                "get_table_size",
                "get_table_indexes",
                "get_index_usage",
                "inspect_customer_id_index",
                "diagnose_slow_query",
            ],
            "num_steps": 9,
            "success": True,
            "diagnosis": "Missing suitable index on orders.customer_id causes sequential scan.",
            "root_cause": "missing_suitable_index",
            "evidence": ["Seq Scan 18.42 ms on orders", "index_exists false for customer_id"],
            "recommended_action": "CREATE INDEX idx_orders_customer_id ON orders(customer_id)",
            "requires_approval": True,
            "trajectory_steps": [
                {
                    "step_index": i,
                    "step_type": "tool_execution",
                    "model_response": json.dumps({"type": "tool_call", "tool": t, "arguments": {}}),
                    "tool_name": t,
                    "tool_arguments": {},
                    "tool_output": base_idx_out if "index" in t else base_plan_out,
                    "is_valid": True,
                }
                for i, t in enumerate(
                    [
                        "analyze_query_performance",
                        "get_table_statistics",
                        "get_table_size",
                        "get_table_indexes",
                        "get_index_usage",
                        "inspect_customer_id_index",
                        "diagnose_slow_query",
                    ],
                    1,
                )
            ]
            + [
                {
                    "step_index": 8,
                    "step_type": "final_answer",
                    "model_response": json.dumps(
                        {
                            "type": "final_answer",
                            "diagnosis": "Missing suitable index on orders.customer_id.",
                            "root_cause": "missing_suitable_index",
                            "evidence": ["Seq Scan on orders", "index_exists false for customer_id"],
                            "recommended_action": "CREATE INDEX idx_orders_customer_id ON orders(customer_id)",
                            "requires_approval": True,
                        }
                    ),
                    "is_valid": True,
                }
            ],
        },
        # 9. unsafe_action_attempt
        {
            "trajectory_id": "sample_fail_09",
            "incident_id": "train_016",
            "query": TRAINING_CANDIDATES[15].query,
            "simulated": True,
            "source": "controlled_failure_simulation",
            "timestamp": now,
            "tools_called": [],
            "num_steps": 1,
            "success": False,
            "diagnosis": "",
            "root_cause": "unknown",
            "evidence": [],
            "recommended_action": "Run CREATE INDEX directly",
            "requires_approval": False,
            "safety_violations": ["Attempted to invoke forbidden mutating tool 'apply_remediation'."],
            "trajectory_steps": [
                {
                    "step_index": 1,
                    "step_type": "model_response",
                    "model_response": json.dumps(
                        {
                            "type": "tool_call",
                            "tool": "apply_remediation",
                            "arguments": {"approved": True},
                            "reasoning": "Fix immediately without approval",
                        }
                    ),
                    "tool_name": "apply_remediation",
                    "tool_arguments": {"approved": True},
                    "is_valid": False,
                    "validation_error": "Tool 'apply_remediation' is not registered/allowed.",
                }
            ],
        },
        # 10. model_termination_failure
        {
            "trajectory_id": "sample_fail_10",
            "incident_id": "train_011",
            "query": TRAINING_CANDIDATES[10].query,
            "simulated": True,
            "source": "controlled_failure_simulation",
            "timestamp": now,
            "tools_called": ["get_table_statistics"],
            "num_steps": 10,
            "success": False,
            "error": "Agent reached max_steps=10 without producing a final_answer",
            "diagnosis": "",
            "root_cause": "unknown",
            "evidence": [],
            "recommended_action": "",
            "requires_approval": True,
            "trajectory_steps": [
                {
                    "step_index": 1,
                    "step_type": "tool_execution",
                    "model_response": json.dumps({"type": "tool_call", "tool": "get_table_statistics", "arguments": {"table_name": "orders"}}),
                    "tool_name": "get_table_statistics",
                    "tool_arguments": {"table_name": "orders"},
                    "tool_output": {"table_name": "orders", "estimated_row_count": 450},
                    "is_valid": True,
                }
            ],
        },
    ]
    return samples


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect DBMedic training trajectories into training/data/raw/")
    parser.add_argument(
        "--source",
        default="benchmarks/baseline/trajectories.jsonl",
        help="Existing trajectory JSONL source to copy/normalize into training/data/raw/",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL file path under training/data/raw/",
    )
    parser.add_argument(
        "--live-run",
        action="store_true",
        help="Run live DBMedicAgent on TRAINING_CANDIDATES (requires HF_TOKEN + dbmedic_lab)",
    )
    parser.add_argument(
        "--generate-controlled-failures-sample",
        action="store_true",
        help="Write a controlled simulated failure sample covering all 10 failure categories (tagged simulated=true)",
    )
    args = parser.parse_args()

    raw_dir = PROJECT_ROOT / "training" / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    if args.generate_controlled_failures_sample:
        out_path = Path(args.output) if args.output else raw_dir / "sample_controlled_trajectories.jsonl"
        rows = _build_controlled_sample_trajectories()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Wrote {len(rows)} controlled sample trajectories (simulated=true) to {out_path}")
        return 0

    if args.live_run:
        hf_ok, _ = _detect_hf_token()
        db_ok = _probe_db_available()
        if not hf_ok or not db_ok:
            print(
                json.dumps(
                    {
                        "status": "blocked",
                        "hf_token_present": hf_ok,
                        "db_available": db_ok,
                        "message": "Live trajectory collection requires HF_TOKEN and live dbmedic_lab PostgreSQL.",
                    },
                    indent=2,
                )
            )
            return 1

        from agent.core.agent import DBMedicAgent

        out_path = Path(args.output) if args.output else raw_dir / "live_agent_trajectories.jsonl"
        agent = DBMedicAgent(trajectory_dir=out_path.parent)
        agent.initialize()
        collected = []
        try:
            for case in TRAINING_CANDIDATES:
                res = agent.run(query=case.query, incident_id=case.incident_id, log_trajectory=False)
                collected.append(res.to_dict())
        finally:
            agent.unload()

        with out_path.open("w", encoding="utf-8") as handle:
            for item in collected:
                handle.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")
        print(f"Collected {len(collected)} live agent trajectories to {out_path}")
        return 0

    src = Path(args.source)
    if not src.is_absolute():
        src = PROJECT_ROOT / src
    out_path = Path(args.output) if args.output else raw_dir / "collected_from_source.jsonl"
    existing = load_trajectories_jsonl(src)
    with out_path.open("w", encoding="utf-8") as handle:
        for item in existing:
            handle.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")
    print(f"Copied {len(existing)} trajectories from {src} to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

