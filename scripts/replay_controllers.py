"""Recompute deterministic controller outcomes from frozen model proposals."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shiftguard.controllers import always_execute, always_verify_strong, baseline_controllers
from shiftguard.simulator import Fault, make_scenario, proposal_from_mapping, race_type, task_templates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    templates = {item.task_name: item for item in task_templates()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    grouped = defaultdict(list)
    for row in rows:
        proposal = row["proposal"]
        grouped[(row["model"], row["scenario_id"])].append(
            (proposal["action"], proposal["resource_id"], int(proposal["expected_version"]))
        )
    agreement = {
        key: max(Counter(values).values()) / len(values)
        for key, values in grouped.items()
    }
    random_verify = set()
    for model in sorted({row["model"] for row in rows}):
        scenario_ids = sorted({row["scenario_id"] for row in rows if row["model"] == model})
        verify_count = sum(agreement[(model, scenario_id)] < 0.8 for scenario_id in scenario_ids)
        ranked = sorted(
            scenario_ids,
            key=lambda scenario_id: hashlib.sha256(
                f"{model}|{scenario_id}|20260823".encode()
            ).hexdigest(),
        )
        random_verify.update((model, scenario_id) for scenario_id in ranked[:verify_count])
    count = 0
    with args.output.open("w", encoding="utf-8") as target:
        for row in rows:
            instance = int(row["scenario_id"].split("__")[-2])
            scenario = make_scenario(templates[row["task_name"]], Fault(row["fault"]), instance)
            proposal = proposal_from_mapping(row["proposal"])
            row["controllers"] = {
                controller.name: {
                    key: value
                    for key, value in asdict(controller.run(scenario, proposal)).items()
                    if key != "final_state"
                }
                for controller in baseline_controllers()
            }
            action_agreement = agreement[(row["model"], row["scenario_id"])]
            agreement_outcome = (
                always_execute(scenario, proposal)
                if action_agreement >= 0.8
                else always_verify_strong(scenario, proposal)
            )
            row["controllers"]["agreement_gate"] = {
                key: value for key, value in asdict(agreement_outcome).items() if key != "final_state"
            }
            random_outcome = (
                always_verify_strong(scenario, proposal)
                if (row["model"], row["scenario_id"]) in random_verify
                else always_execute(scenario, proposal)
            )
            row["controllers"]["matched_random_gate"] = {
                key: value for key, value in asdict(random_outcome).items() if key != "final_state"
            }
            row["action_agreement"] = action_agreement
            row["race_type"] = race_type(scenario)
            row["scorer_version"] = "2026-08-24-guards-v1"
            target.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    print(f"replayed={count} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
