"""Produce compact CSV/Markdown summaries without third-party dependencies."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def mean(values):
    return sum(values) / len(values) if values else float("nan")


def main() -> int:
    args = parse_args()
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    grouped = defaultdict(list)
    for row in rows:
        for controller, outcome in row["controllers"].items():
            grouped[(row["model"], controller)].append((row, outcome))

    summary = []
    for (model, controller), items in sorted(grouped.items()):
        summary.append(
            {
                "model": model,
                "controller": controller,
                "n": len(items),
                "valid_json": mean([float(row["proposal"]["valid_json"]) for row, _ in items]),
                "unsafe_rate": mean([float(outcome["unsafe"]) for _, outcome in items]),
                "safe_success": mean([float(outcome["safe_task_success"]) for _, outcome in items]),
                "verification_cost": mean([float(outcome["verification_cost"]) for _, outcome in items]),
                "escalation_rate": mean([float(not outcome["automated"]) for _, outcome in items]),
                "mean_confidence": mean([float(row["proposal"]["confidence"]) for row, _ in items]),
                "model_seconds": mean([float(row["model_seconds"]) for row, _ in items]),
            }
        )

    fieldnames = list(summary[0]) if summary else []
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary)

    print("| model | controller | n | valid JSON | unsafe | safe success | verify cost | escalation | mean conf | sec |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in summary:
        print(
            f"| {row['model']} | {row['controller']} | {row['n']} | "
            f"{row['valid_json']:.3f} | {row['unsafe_rate']:.3f} | "
            f"{row['safe_success']:.3f} | {row['verification_cost']:.2f} | "
            f"{row['escalation_rate']:.3f} | {row['mean_confidence']:.3f} | "
            f"{row['model_seconds']:.2f} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

