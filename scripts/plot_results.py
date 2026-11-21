"""Create paper-ready PDF figures from replayed trajectories and CSV metrics."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


DISPLAY = {
    "qwen3:4b": "Qwen3-4B",
    "phi4-mini:latest": "Phi-4-mini",
    "gemma4:latest": "Gemma4-8B",
    "always_execute": "Execute",
    "always_verify_fast": "Fast verify",
    "always_verify_strong": "Strong verify",
    "evidence_router": "Evidence router",
}
COLORS = {
    "always_execute": "#D55E00",
    "always_verify_fast": "#E69F00",
    "always_verify_strong": "#0072B2",
    "evidence_router": "#009E73",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_csv(path):
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def reliability(rows, bins=10):
    result = []
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        members = []
        for row in rows:
            confidence = float(row["proposal"]["confidence"])
            if lower <= confidence < upper or (index == bins - 1 and confidence == 1):
                outcome = row["controllers"]["always_execute"]
                correct = float(not outcome["unsafe"] and outcome["safe_task_success"])
                members.append((confidence, correct))
        if members:
            result.append((sum(x[0] for x in members) / len(members), sum(x[1] for x in members) / len(members), len(members)))
    return result


def main() -> int:
    args = parse_args()
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "pdf.fonttype": 42})

    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.35), sharex=True, sharey=True)
    for axis, model in zip(axes, sorted({row["model"] for row in rows})):
        points = reliability([row for row in rows if row["model"] == model])
        axis.plot([0, 1], [0, 1], "--", color="0.6", linewidth=1)
        axis.plot([x[0] for x in points], [x[1] for x in points], "o-", color="#0072B2")
        for confidence, accuracy, count in points:
            axis.annotate(str(count), (confidence, accuracy), xytext=(3, 3), textcoords="offset points", fontsize=6)
        axis.set_title(DISPLAY[model])
        axis.set_xlim(0, 1.02)
        axis.set_ylim(0, 1.02)
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Empirical safe correctness")
    axes[1].set_xlabel("Verbal confidence")
    fig.tight_layout()
    fig.savefig(args.output_dir / "confidence_reliability.pdf", bbox_inches="tight")
    plt.close(fig)

    profile = read_csv(args.analysis_dir / "profile_metrics.csv")
    controllers = ["always_execute", "always_verify_fast", "always_verify_strong", "evidence_router"]
    profiles = sorted({row["profile"] for row in profile})
    fig, axes = plt.subplots(1, len(profiles), figsize=(3.55 * len(profiles), 2.5), squeeze=False)
    axes = axes[0]
    for axis, profile_name in zip(axes, profiles):
        subset = [row for row in profile if row["profile"] == profile_name]
        for controller in controllers:
            points = [row for row in subset if row["controller"] == controller]
            if not points:
                continue
            unsafe = sum(float(row["unsafe_rate"]) for row in points) / len(points)
            cost = sum(float(row["verification_cost"]) for row in points) / len(points)
            axis.scatter(cost, unsafe, s=45, color=COLORS[controller], label=DISPLAY[controller], zorder=3)
            axis.annotate(DISPLAY[controller], (cost, unsafe), xytext=(3, 3), textcoords="offset points", fontsize=6)
        axis.set_title(profile_name.capitalize() + " telemetry")
        axis.set_xlabel("Mean verification cost")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Unsafe commit rate")
    fig.tight_layout()
    fig.savefig(args.output_dir / "risk_cost_profiles.pdf", bbox_inches="tight")
    plt.close(fig)

    controller_rows = read_csv(args.analysis_dir / "controller_metrics.csv")
    if controller_rows and "benign_block_rate" in controller_rows[0]:
        frontier = {
            "always_execute": ("Execute", "#D55E00", "o"),
            "always_verify_fast": ("Fast verify", "#E69F00", "o"),
            "always_verify_strong": ("Strong verify", "#0072B2", "o"),
            "evidence_router": ("Evidence router", "#009E73", "o"),
            "global_epoch_guard": ("Epoch guard", "#CC79A7", "D"),
            "read_set_version_guard": ("Version guard", "#CC79A7", "s"),
            "commit_predicate_guard": ("Predicate guard", "#009E73", "*"),
            "predicate_guard_no_policy": ("no policy", "#D55E00", "x"),
            "predicate_guard_no_operational": ("no status", "#D55E00", "x"),
            "predicate_guard_no_duplicate": ("no duplicate", "#D55E00", "x"),
        }
        grouped = defaultdict(list)
        for row in controller_rows:
            grouped[row["controller"]].append(row)
        fig, axis = plt.subplots(figsize=(3.55, 2.6))
        for name, (label, color, marker) in frontier.items():
            points = grouped.get(name)
            if not points:
                continue
            benign = sum(float(row["benign_block_rate"]) for row in points) / len(points)
            unsafe = sum(float(row["unsafe_rate"]) for row in points) / len(points)
            size = 90 if marker == "*" else 40
            axis.scatter(benign, unsafe, s=size, color=color, marker=marker, zorder=3)
            axis.annotate(label, (benign, unsafe), xytext=(4, 3), textcoords="offset points", fontsize=6)
        axis.set_xlabel("Benign intervention rate (availability loss)")
        axis.set_ylabel("Unsafe commit rate")
        axis.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(args.output_dir / "guard_frontier.pdf", bbox_inches="tight")
        plt.close(fig)

    print(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
