"""Render measured CSV metrics as compact LaTeX tables."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


MODEL = {
    "qwen3:4b": "Qwen3-4B",
    "phi4-mini:latest": "Phi-4-mini",
    "gemma4:latest": "Gemma4-8B",
    "llama3.2:3b": "Llama-3.2-3B",
}
CONTROL = {
    "always_execute": "Execute",
    "always_verify_fast": "Fast verify",
    "always_verify_strong": "Strong verify",
    "confidence_gate": "Confidence gate",
    "agreement_gate": "Agreement gate",
    "matched_random_gate": "Matched random",
    "ttl_gate": "TTL gate",
    "evidence_router": "Evidence router",
    "global_epoch_guard": "Epoch guard",
    "read_set_version_guard": "Version guard",
    "commit_predicate_guard": "Predicate guard",
}
GUARDS = {
    "global_epoch_guard": "Global epoch",
    "read_set_version_guard": "Read-set version",
    "commit_predicate_guard": "Commit predicate",
}
DROPS = {
    "commit_predicate_guard": ("Complete contract", "--"),
    "predicate_guard_no_integrity": ("No integrity clause", "none (subsumed)"),
    "predicate_guard_no_duplicate": ("No duplicate clause", "completed effects"),
    "predicate_guard_no_operational": ("No status clause", "stale reads"),
    "predicate_guard_no_policy": ("No policy clause", "concurrent mutation"),
}
LEVELS = {"fast": "Fast verify", "strong": "Strong verify", "guard_block": "Guard block + refresh"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--router-metrics", type=Path, required=True)
    parser.add_argument("--layer-b-summary", type=Path, default=None)
    parser.add_argument("--aware-analysis-dir", type=Path, default=None)
    parser.add_argument("--guard-cost-metrics", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load(path):
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def avg(items, key):
    return sum(float(item[key]) for item in items) / len(items)


def main() -> int:
    args = parse_args()
    proposal = load(args.analysis_dir / "proposal_metrics.csv")
    controller = load(args.analysis_dir / "controller_metrics.csv")
    profile = load(args.analysis_dir / "profile_metrics.csv")
    router = load(args.router_metrics)
    lines = [
        "% Generated from measured artifacts; do not hand-edit.",
        "\\begin{table}[t]",
        "\\caption{Main proposal metrics. AUROC is confidence predicting correct direct execution.}",
        "\\label{tab:mainconf}",
        "\\centering\\small",
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Model & Valid & Abstain & Unsafe & AUROC & ECE \\\\",
        "\\midrule",
    ]
    for item in sorted(proposal, key=lambda row: row["model"]):
        lines.append(
            f"{MODEL[item['model']]} & {float(item['valid_json']):.3f} & {float(item['abstain_rate']):.3f} & "
            f"{float(item['direct_unsafe']):.3f} & {float(item['confidence_auroc_correct']):.3f} & "
            f"{float(item['confidence_ece10']):.3f} \\\\"
        )
    lines.extend([
        "\\bottomrule", "\\end{tabular}", "\\end{table}", "",
        "\\begin{table}[t]",
        "\\caption{Controller macro-average over the three model families. Cost uses normalized verification units.}",
        "\\label{tab:maincontrollers}",
        "\\centering\\small",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{lrrrr}",
        "\\toprule", "Controller & Unsafe & Safe succ. & Benign block & Verify cost \\\\", "\\midrule",
    ])
    for name in CONTROL:
        items = [row for row in controller if row["controller"] == name]
        if not items:
            continue
        lines.append(
            f"{CONTROL[name]} & {avg(items, 'unsafe_rate'):.3f} & {avg(items, 'safe_success'):.3f} & "
            f"{avg(items, 'benign_block_rate'):.3f} & {avg(items, 'verification_cost'):.2f} \\\\"
        )
    lines.extend([
        "\\bottomrule", "\\end{tabular}}", "\\end{table}", "",
        "\\begin{table}[t]",
        "\\caption{Evidence router under telemetry shift (macro-average over models).}",
        "\\label{tab:profiles}",
        "\\centering\\small",
        "\\begin{tabular}{lrrr}",
        "\\toprule", "Profile & Unsafe & Safe succ. & Verify cost \\\\", "\\midrule",
    ])
    for profile_name in ["canonical", "degraded"]:
        items = [row for row in profile if row["profile"] == profile_name and row["controller"] == "evidence_router"]
        if items:
            lines.append(
                f"{profile_name.capitalize()} & {avg(items, 'unsafe_rate'):.3f} & {avg(items, 'safe_success'):.3f} & "
                f"{avg(items, 'verification_cost'):.2f} \\\\"
            )
    lines.extend([
        "\\bottomrule", "\\end{tabular}", "\\end{table}", "",
        "\\begin{table}[t]",
        "\\caption{Learned support-aware router on frozen test splits.}",
        "\\label{tab:learnedrouter}",
        "\\centering\\small",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{llrrr}",
        "\\toprule", "Split & Model & Unsafe & Cost & Unsupported \\\\", "\\midrule",
    ])
    split_names = {"id_heldout_template": "ID template", "ood_degraded_telemetry": "OOD telemetry"}
    for split in split_names:
        for item in sorted([row for row in router if row["split"] == split], key=lambda row: row["model"]):
            lines.append(
                f"{split_names[split]} & {MODEL[item['model']]} & {float(item['unsafe_rate']):.3f} & "
                f"{float(item['verification_cost']):.2f} & {float(item['unsupported_rate']):.3f} \\\\"
            )
    lines.extend(["\\bottomrule", "\\end{tabular}}", "\\end{table}", ""])

    race_path = args.analysis_dir / "race_metrics.csv"
    if race_path.exists():
        race = load(race_path)

        def race_cell(name, label, key):
            items = [
                row for row in race
                if row["controller"] == name and row["race_type"] == label
            ]
            return avg(items, key) if items else float("nan")

        lines.extend([
            "\\begin{table}[t]",
            "\\caption{Guard precision by race type, pooled over models. Block rates on",
            "invalidating races enforce safety; blocks on predicate-preserving or",
            "irrelevant races are benign interventions that forfeit the task.}",
            "\\label{tab:guardprecision}",
            "\\centering\\small",
            "\\resizebox{\\columnwidth}{!}{%",
            "\\begin{tabular}{lrrrr}",
            "\\toprule",
            "Guard & Unsafe & Block\\textsubscript{inv} & Benign\\textsubscript{pres} & Benign\\textsubscript{irr} \\\\",
            "\\midrule",
        ])
        for name, label in GUARDS.items():
            lines.append(
                f"{label} & {race_cell(name, 'invalidating', 'unsafe_rate'):.3f} & "
                f"{race_cell(name, 'invalidating', 'block_rate'):.3f} & "
                f"{race_cell(name, 'predicate_preserving', 'benign_block_rate'):.3f} & "
                f"{race_cell(name, 'irrelevant', 'benign_block_rate'):.3f} \\\\"
            )
        lines.extend([
            "\\bottomrule", "\\end{tabular}}", "\\end{table}", "",
            "\\begin{table}[t]",
            "\\caption{Predicate guard under contract incompleteness (macro-average over",
            "models). Each missing clause converts exactly its fault family into unsafe",
            "commits without touching availability.}",
            "\\label{tab:incompleteness}",
            "\\centering\\small",
            "\\resizebox{\\columnwidth}{!}{%",
            "\\begin{tabular}{lrrl}",
            "\\toprule", "Declared contract & Unsafe & Safe succ. & Leaked hazards \\\\", "\\midrule",
        ])
        for name, (label, leak) in DROPS.items():
            items = [row for row in controller if row["controller"] == name]
            if not items:
                continue
            lines.append(
                f"{label} & {avg(items, 'unsafe_rate'):.3f} & "
                f"{avg(items, 'safe_success'):.3f} & {leak} \\\\"
            )
        lines.extend(["\\bottomrule", "\\end{tabular}}", "\\end{table}", ""])

    if args.guard_cost_metrics and args.guard_cost_metrics.exists():
        guard_cost = [
            row for row in load(args.guard_cost_metrics) if row["model"] == "macro"
        ]
        lines.extend([
            "\\begin{table}[t]",
            "\\caption{Predicate-check cost sensitivity at $\\lambda_U=100$ and",
            "$\\lambda_F=10$ (macro-average). Outcomes are fixed; only the normalized",
            "semantic-check cost $c_p$ varies.}",
            "\\label{tab:guardcost}",
            "\\centering\\small",
            "\\begin{tabular}{rrrr}",
            "\\toprule",
            "$c_p$ & Predicate & Version & Strong verify \\\\",
            "\\midrule",
        ])
        for item in guard_cost:
            lines.append(
                f"{float(item['guard_cost']):g} & {float(item['predicate_loss']):.3f} & "
                f"{float(item['version_loss']):.3f} & {float(item['strong_loss']):.3f} \\\\"
            )
        lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])

    if args.layer_b_summary and args.layer_b_summary.exists():
        layer_b = load(args.layer_b_summary)
        lines.extend([
            "\\begin{table}[t]",
            "\\caption{Layer-B replanning after returned evidence, on the matched",
            "population of race decisions where a version guard fires (432 per level,",
            "pooled over models). Recommit-unsafe is the unsafe rate among",
            "non-abstaining replans.}",
            "\\label{tab:layerb}",
            "\\centering\\small",
            "\\resizebox{\\columnwidth}{!}{%",
            "\\begin{tabular}{lrrr}",
            "\\toprule", "Returned evidence & Abstain & Recommit & Recommit unsafe \\\\", "\\midrule",
        ])
        for level, label in LEVELS.items():
            items = [row for row in layer_b if row["verification_level"] == level]
            if not items:
                continue
            lines.append(
                f"{label} & {avg(items, 'abstain_rate'):.3f} & "
                f"{avg(items, 'recommit_rate'):.3f} & {avg(items, 'recommit_unsafe_rate'):.3f} \\\\"
            )
        lines.extend(["\\bottomrule", "\\end{tabular}}", "\\end{table}", ""])

    if args.aware_analysis_dir and (args.aware_analysis_dir / "proposal_metrics.csv").exists():
        aware = load(args.aware_analysis_dir / "proposal_metrics.csv")
        lines.extend([
            "\\begin{table}[t]",
            "\\caption{Prompt-level caution control: identical scenarios with an explicit",
            "instruction to consider staleness, ambiguous commits, and concurrency.}",
            "\\label{tab:aware}",
            "\\centering\\small",
            "\\begin{tabular}{lrrrr}",
            "\\toprule",
            "Model & \\multicolumn{2}{c}{Abstain} & \\multicolumn{2}{c}{Direct unsafe} \\\\",
            " & base & aware & base & aware \\\\",
            "\\midrule",
        ])
        for item in sorted(aware, key=lambda row: row["model"]):
            base_items = [row for row in proposal if row["model"] == item["model"]]
            if not base_items:
                continue
            base = base_items[0]
            lines.append(
                f"{MODEL[item['model']]} & {float(base['abstain_rate']):.3f} & "
                f"{float(item['abstain_rate']):.3f} & {float(base['direct_unsafe']):.3f} & "
                f"{float(item['direct_unsafe']):.3f} \\\\"
            )
        lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
