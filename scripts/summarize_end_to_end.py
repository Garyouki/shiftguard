"""Summarize verification-conditioned replanning trajectories."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--matched-output", type=Path, default=None,
        help="Also write per-level metrics restricted to the guard-block-eligible "
        "decisions, so all levels are compared on the same population.")
    return parser.parse_args()


def mean(values):
    return sum(values) / len(values) if values else float("nan")


def main() -> int:
    args = parse_args()
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["telemetry_profile"], row["verification_level"])].append(row)
    summary = []
    for (model, profile, level), items in sorted(grouped.items()):
        recommits = [row for row in items if row["replanned_proposal"]["action"] != "abstain"]
        summary.append({
            "model": model, "profile": profile, "verification_level": level, "n": len(items),
            "valid_json": mean([float(row["replanned_proposal"]["valid_json"]) for row in items]),
            "abstain_rate": mean([float(row["replanned_proposal"]["action"] == "abstain") for row in items]),
            "unsafe_rate": mean([float(row["outcome"]["unsafe"]) for row in items]),
            "recommit_rate": len(recommits) / len(items) if items else float("nan"),
            "recommit_unsafe_rate": mean([float(row["outcome"]["unsafe"]) for row in recommits]),
            "safe_success": mean([float(row["outcome"]["safe_task_success"]) for row in items]),
            "mean_seconds": mean([float(row["model_seconds"]) for row in items]),
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    for row in summary:
        print(row)

    if args.matched_output:
        eligible = {
            (row["model"], row["scenario_id"], row["seed"])
            for row in rows if row["verification_level"] == "guard_block"
        }
        matched = [
            row for row in rows
            if (row["model"], row["scenario_id"], row["seed"]) in eligible
        ]
        by_level = defaultdict(list)
        for row in matched:
            by_level[row["verification_level"]].append(row)
        matched_summary = []
        for level, items in sorted(by_level.items()):
            recommits = [row for row in items if row["replanned_proposal"]["action"] != "abstain"]
            matched_summary.append({
                "verification_level": level, "n": len(items),
                "abstain_rate": 1 - len(recommits) / len(items),
                "unsafe_rate": mean([float(row["outcome"]["unsafe"]) for row in items]),
                "recommit_rate": len(recommits) / len(items),
                "recommit_unsafe_rate": mean([float(row["outcome"]["unsafe"]) for row in recommits]),
                "safe_success": mean([float(row["outcome"]["safe_task_success"]) for row in items]),
                "mean_seconds": mean([float(row["model_seconds"]) for row in items]),
            })
        with args.matched_output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(matched_summary[0]))
            writer.writeheader()
            writer.writerows(matched_summary)
        for row in matched_summary:
            print("matched", row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
