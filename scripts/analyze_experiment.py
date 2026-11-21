"""Analyze calibration, agreement, safety, and cost from JSONL trajectories."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--unsafe-cost", type=float, default=100.0)
    parser.add_argument("--failure-cost", type=float, default=10.0)
    return parser.parse_args()


def mean(values):
    return sum(values) / len(values) if values else float("nan")


def wilson(successes: int, n: int, z: float = 1.96):
    if not n:
        return (float("nan"), float("nan"))
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return center - radius, center + radius


def clustered_bootstrap_mean(items, value_fn, cluster_fn, repetitions=2000, seed=20260823):
    """Percentile interval after resampling task-template clusters."""

    grouped = defaultdict(list)
    for item in items:
        grouped[cluster_fn(item)].append(item)
    keys = sorted(grouped)
    if len(keys) < 2:
        return (float("nan"), float("nan"))
    generator = random.Random(seed)
    estimates = []
    for _ in range(repetitions):
        sample = []
        for _ in keys:
            sample.extend(grouped[generator.choice(keys)])
        estimates.append(mean([value_fn(item) for item in sample]))
    estimates.sort()
    return estimates[int(0.025 * repetitions)], estimates[int(0.975 * repetitions) - 1]


def auroc(labels, scores):
    positives = [score for label, score in zip(labels, scores) if label]
    negatives = [score for label, score in zip(labels, scores) if not label]
    if not positives or not negatives:
        return float("nan")
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            wins += float(positive > negative) + 0.5 * float(positive == negative)
    return wins / (len(positives) * len(negatives))


def ece(labels, probabilities, bins=10):
    total = len(labels)
    result = 0.0
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        members = [
            (label, probability)
            for label, probability in zip(labels, probabilities)
            if lower <= probability < upper or (index == bins - 1 and probability == 1.0)
        ]
        if members:
            result += len(members) / total * abs(mean([x[1] for x in members]) - mean([x[0] for x in members]))
    return result


def action_key(row):
    proposal = row["proposal"]
    return proposal["action"], proposal["resource_id"], int(proposal["expected_version"])


def benign_block(row, controller):
    """A block counts as benign intervention when direct execution would have
    committed safely and completed the task."""

    outcome = row["controllers"][controller]
    counterfactual = row["controllers"]["always_execute"]
    return (
        not outcome["executed"]
        and counterfactual["executed"]
        and not counterfactual["unsafe"]
        and counterfactual["safe_task_success"]
    )


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    seed_groups = defaultdict(list)
    cross_model_groups = defaultdict(list)
    for row in rows:
        seed_groups[(row["model"], row["scenario_id"])].append(row)
        cross_model_groups[(row["scenario_id"], row["seed"])].append(row)
    seed_agreement = {
        key: max(Counter(action_key(row) for row in items).values()) / len(items)
        for key, items in seed_groups.items()
    }
    model_agreement = {
        key: max(Counter(action_key(row) for row in items).values()) / len(items)
        for key, items in cross_model_groups.items()
    }

    proposal_metrics = []
    for model in sorted({row["model"] for row in rows}):
        items = [row for row in rows if row["model"] == model]
        labels = []
        probabilities = []
        unsafe_labels = []
        seed_scores = []
        model_scores = []
        for row in items:
            outcome = row["controllers"]["always_execute"]
            correct = float(not outcome["unsafe"] and outcome["safe_task_success"])
            labels.append(correct)
            unsafe_labels.append(float(outcome["unsafe"]))
            probabilities.append(float(row["proposal"]["confidence"]))
            seed_scores.append(1.0 - seed_agreement[(model, row["scenario_id"])])
            model_scores.append(1.0 - model_agreement[(row["scenario_id"], row["seed"])])
        proposal_metrics.append({
            "model": model,
            "n": len(items),
            "valid_json": mean([float(row["proposal"]["valid_json"]) for row in items]),
            "abstain_rate": mean([float(row["proposal"]["action"] == "abstain") for row in items]),
            "mean_confidence": mean(probabilities),
            "direct_unsafe": mean(unsafe_labels),
            "direct_safe_success": mean(labels),
            "confidence_auroc_correct": auroc(labels, probabilities),
            "confidence_auroc_unsafe": auroc(unsafe_labels, [1 - value for value in probabilities]),
            "confidence_brier": mean([(probability - label) ** 2 for label, probability in zip(labels, probabilities)]),
            "confidence_ece10": ece(labels, probabilities),
            "seed_disagreement_auroc_unsafe": auroc(unsafe_labels, seed_scores),
            "cross_model_disagreement_auroc_unsafe": auroc(unsafe_labels, model_scores),
            "mean_model_seconds": mean([float(row["model_seconds"]) for row in items]),
        })

    proposal_profile_metrics = []
    for model in sorted({row["model"] for row in rows}):
        for profile in sorted({row.get("telemetry_profile", "canonical") for row in rows}):
            items = [
                row for row in rows
                if row["model"] == model and row.get("telemetry_profile", "canonical") == profile
            ]
            labels = [float(row["controllers"]["always_execute"]["unsafe"]) for row in items]
            confidence = [float(row["proposal"]["confidence"]) for row in items]
            proposal_profile_metrics.append({
                "model": model,
                "profile": profile,
                "n": len(items),
                "abstain_rate": mean([float(row["proposal"]["action"] == "abstain") for row in items]),
                "mean_confidence": mean(confidence),
                "direct_unsafe": mean(labels),
                "risk_auroc": auroc(labels, [1 - value for value in confidence]),
            })

    controller_metrics = []
    for model in sorted({row["model"] for row in rows}):
        items = [row for row in rows if row["model"] == model]
        for controller in sorted(items[0]["controllers"]):
            outcomes = [row["controllers"][controller] for row in items]
            unsafe_count = sum(bool(outcome["unsafe"]) for outcome in outcomes)
            unsafe_low, unsafe_high = clustered_bootstrap_mean(
                items,
                lambda row: float(row["controllers"][controller]["unsafe"]),
                lambda row: row["task_name"],
            )
            costs = [
                args.unsafe_cost * float(outcome["unsafe"])
                + args.failure_cost * float(not outcome["safe_task_success"])
                + float(outcome["verification_cost"])
                + float(outcome["escalation_cost"])
                for outcome in outcomes
            ]
            controller_metrics.append({
                "model": model,
                "controller": controller,
                "n": len(outcomes),
                "unsafe_rate": unsafe_count / len(outcomes),
                "unsafe_ci_low": unsafe_low,
                "unsafe_ci_high": unsafe_high,
                "safe_success": mean([float(outcome["safe_task_success"]) for outcome in outcomes]),
                "benign_block_rate": mean([float(benign_block(row, controller)) for row in items]),
                "verification_cost": mean([float(outcome["verification_cost"]) for outcome in outcomes]),
                "escalation_rate": mean([float(not outcome["automated"]) for outcome in outcomes]),
                "expected_loss": mean(costs),
            })

    race_metrics = []
    race_labels = sorted({row.get("race_type", "unknown") for row in rows})
    if race_labels != ["unknown"]:
        for controller in sorted(rows[0]["controllers"]):
            for label in race_labels:
                items = [row for row in rows if row.get("race_type") == label]
                if not items:
                    continue
                outcomes = [row["controllers"][controller] for row in items]
                race_metrics.append({
                    "controller": controller,
                    "race_type": label,
                    "n": len(items),
                    "block_rate": mean([float(not outcome["executed"]) for outcome in outcomes]),
                    "unsafe_rate": mean([float(outcome["unsafe"]) for outcome in outcomes]),
                    "benign_block_rate": mean([float(benign_block(row, controller)) for row in items]),
                    "safe_success": mean([float(outcome["safe_task_success"]) for outcome in outcomes]),
                })

    fault_metrics = []
    for fault in sorted({row["fault"] for row in rows}):
        for controller in ["always_execute", "always_verify_fast", "always_verify_strong", "evidence_router"]:
            items = [row["controllers"][controller] for row in rows if row["fault"] == fault]
            fault_metrics.append({
                "fault": fault,
                "controller": controller,
                "n": len(items),
                "unsafe_rate": mean([float(item["unsafe"]) for item in items]),
                "safe_success": mean([float(item["safe_task_success"]) for item in items]),
                "verification_cost": mean([float(item["verification_cost"]) for item in items]),
            })

    profile_metrics = []
    profiles = sorted({row.get("telemetry_profile", "canonical") for row in rows})
    for profile in profiles:
        for model in sorted({row["model"] for row in rows}):
            selected = [
                row for row in rows
                if row["model"] == model and row.get("telemetry_profile", "canonical") == profile
            ]
            for controller in ["always_execute", "always_verify_fast", "always_verify_strong", "evidence_router"]:
                outcomes = [row["controllers"][controller] for row in selected]
                profile_metrics.append({
                    "profile": profile,
                    "model": model,
                    "controller": controller,
                    "n": len(outcomes),
                    "unsafe_rate": mean([float(item["unsafe"]) for item in outcomes]),
                    "safe_success": mean([float(item["safe_task_success"]) for item in outcomes]),
                    "verification_cost": mean([float(item["verification_cost"]) for item in outcomes]),
                })

    write_csv(args.output_dir / "proposal_metrics.csv", proposal_metrics)
    write_csv(args.output_dir / "proposal_profile_metrics.csv", proposal_profile_metrics)
    write_csv(args.output_dir / "controller_metrics.csv", controller_metrics)
    write_csv(args.output_dir / "fault_metrics.csv", fault_metrics)
    write_csv(args.output_dir / "profile_metrics.csv", profile_metrics)
    write_csv(args.output_dir / "race_metrics.csv", race_metrics)
    (args.output_dir / "metrics.json").write_text(
        json.dumps({"proposal": proposal_metrics, "proposal_profile": proposal_profile_metrics, "controller": controller_metrics, "fault": fault_metrics, "profile": profile_metrics, "race": race_metrics}, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Experiment summary",
        "",
        "## Proposal and confidence metrics",
        "",
        "| Model | N | Valid | Abstain | Direct unsafe | Safe success | AUROC(conf→correct) | ECE | AUROC(seed disagreement→unsafe) | AUROC(model disagreement→unsafe) | Model sec |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in proposal_metrics:
        lines.append(
            f"| {item['model']} | {item['n']} | {item['valid_json']:.3f} | {item['abstain_rate']:.3f} | "
            f"{item['direct_unsafe']:.3f} | {item['direct_safe_success']:.3f} | "
            f"{item['confidence_auroc_correct']:.3f} | {item['confidence_ece10']:.3f} | "
            f"{item['seed_disagreement_auroc_unsafe']:.3f} | {item['cross_model_disagreement_auroc_unsafe']:.3f} | "
            f"{item['mean_model_seconds']:.2f} |"
        )
    lines.extend([
        "",
        "## Controller metrics",
        "",
        f"Expected loss uses unsafe cost={args.unsafe_cost:g}, failed safe completion cost={args.failure_cost:g}.",
        "",
        "| Model | Controller | Unsafe | 95% CI | Safe success | Benign block | Verify cost | Escalation | Expected loss |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for item in controller_metrics:
        lines.append(
            f"| {item['model']} | {item['controller']} | {item['unsafe_rate']:.3f} | "
            f"[{item['unsafe_ci_low']:.3f}, {item['unsafe_ci_high']:.3f}] | {item['safe_success']:.3f} | "
            f"{item['benign_block_rate']:.3f} | "
            f"{item['verification_cost']:.2f} | {item['escalation_rate']:.3f} | {item['expected_loss']:.2f} |"
        )
    if race_metrics:
        lines.extend([
            "",
            "## Guard precision by race type (pooled across models)",
            "",
            "Benign block: the controller withheld a commit that direct execution",
            "would have completed safely.",
            "",
            "| Controller | Race type | N | Block | Unsafe | Benign block | Safe success |",
            "|---|---|---:|---:|---:|---:|---:|",
        ])
        for item in race_metrics:
            lines.append(
                f"| {item['controller']} | {item['race_type']} | {item['n']} | "
                f"{item['block_rate']:.3f} | {item['unsafe_rate']:.3f} | "
                f"{item['benign_block_rate']:.3f} | {item['safe_success']:.3f} |"
            )
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
