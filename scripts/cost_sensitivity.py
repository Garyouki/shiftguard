"""Scan unsafe/failure costs for every replayed controller."""

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
    parser.add_argument("--unsafe-costs", nargs="+", type=float, default=[10, 25, 50, 100, 250])
    parser.add_argument("--failure-costs", nargs="+", type=float, default=[1, 10])
    parser.add_argument(
        "--guard-cost-output",
        type=Path,
        default=None,
        help="Optional CSV for predicate-guard cost sensitivity.",
    )
    parser.add_argument(
        "--predicate-guard-costs",
        nargs="+",
        type=float,
        default=[1, 1.5, 2, 3, 4],
    )
    return parser.parse_args()


def mean(values):
    return sum(values) / len(values)


def main() -> int:
    args = parse_args()
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    grouped = defaultdict(list)
    for row in rows:
        for controller, outcome in row["controllers"].items():
            grouped[(row["model"], controller)].append(outcome)
    results = []
    for unsafe_cost in args.unsafe_costs:
        for failure_cost in args.failure_costs:
            for (model, controller), outcomes in sorted(grouped.items()):
                loss = mean([
                    unsafe_cost * float(item["unsafe"])
                    + failure_cost * float(not item["safe_task_success"])
                    + float(item["verification_cost"])
                    + float(item["escalation_cost"])
                    for item in outcomes
                ])
                results.append({
                    "unsafe_cost": unsafe_cost,
                    "failure_cost": failure_cost,
                    "model": model,
                    "controller": controller,
                    "expected_loss": loss,
                })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    if args.guard_cost_output:
        # Keep the measured outcomes fixed and vary only the normalized price
        # of evaluating the semantic predicate at commit time.
        unsafe_cost = 100.0
        failure_cost = 10.0
        guard_rows = []
        models = sorted({model for model, _ in grouped})

        def loss(outcomes, verification_override=None):
            return mean([
                unsafe_cost * float(item["unsafe"])
                + failure_cost * float(not item["safe_task_success"])
                + (
                    float(verification_override)
                    if verification_override is not None
                    else float(item["verification_cost"])
                )
                + float(item["escalation_cost"])
                for item in outcomes
            ])

        for guard_cost in args.predicate_guard_costs:
            per_model = []
            for model in models:
                predicate_loss = loss(
                    grouped[(model, "commit_predicate_guard")],
                    verification_override=guard_cost,
                )
                version_loss = loss(grouped[(model, "read_set_version_guard")])
                strong_loss = loss(grouped[(model, "always_verify_strong")])
                values = {
                    "Predicate guard": predicate_loss,
                    "Version guard": version_loss,
                    "Strong verify": strong_loss,
                }
                row = {
                    "guard_cost": guard_cost,
                    "model": model,
                    "predicate_loss": predicate_loss,
                    "version_loss": version_loss,
                    "strong_loss": strong_loss,
                    "best": min(values, key=values.get),
                }
                guard_rows.append(row)
                per_model.append(row)
            macro_values = {
                key: mean([float(row[key]) for row in per_model])
                for key in ("predicate_loss", "version_loss", "strong_loss")
            }
            labels = {
                "Predicate guard": macro_values["predicate_loss"],
                "Version guard": macro_values["version_loss"],
                "Strong verify": macro_values["strong_loss"],
            }
            guard_rows.append({
                "guard_cost": guard_cost,
                "model": "macro",
                **macro_values,
                "best": min(labels, key=labels.get),
            })
        args.guard_cost_output.parent.mkdir(parents=True, exist_ok=True)
        with args.guard_cost_output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(guard_rows[0]))
            writer.writeheader()
            writer.writerows(guard_rows)
        print(f"guard_cost_output={args.guard_cost_output}")
    winners = defaultdict(list)
    for item in results:
        winners[(item["unsafe_cost"], item["failure_cost"], item["model"])].append(item)
    for key, items in sorted(winners.items()):
        best = min(items, key=lambda item: item["expected_loss"])
        print(f"unsafe={key[0]:g} failure={key[1]:g} model={key[2]} best={best['controller']} loss={best['expected_loss']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
