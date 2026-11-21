"""Exhaustively validate safety invariants and expected blind spots."""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shiftguard.controllers import baseline_controllers
from shiftguard.simulator import Proposal, generate_scenarios


def main() -> int:
    scenarios = generate_scenarios(instances_per_template=4)
    controllers = baseline_controllers()
    grouped = defaultdict(Counter)
    failures = []
    for scenario in scenarios:
        proposal = Proposal(
            scenario.template.operation, scenario.resource_id,
            int(scenario.visible_state["version"]), 0.95,
        )
        for controller in controllers:
            outcome = controller.run(scenario, proposal)
            key = (scenario.telemetry_profile, controller.name)
            grouped[key]["n"] += 1
            grouped[key]["unsafe"] += int(outcome.unsafe)
            grouped[key]["safe_success"] += int(outcome.safe_task_success)
            grouped[key]["verification_cost"] += outcome.verification_cost
            if controller.name == "always_verify_strong" and outcome.unsafe:
                failures.append(f"strong verifier unsafe: {scenario.scenario_id}")
    expected_hidden = {
        item.scenario_id for item in scenarios
        if item.telemetry_profile == "degraded" and item.hazardous and item.fault.value == "concurrent_mutation"
    }
    observed_hidden = set()
    shift = next(item for item in controllers if item.name == "evidence_router")
    for scenario in scenarios:
        proposal = Proposal(scenario.template.operation, scenario.resource_id, int(scenario.visible_state["version"]), 0.95)
        if shift.run(scenario, proposal).unsafe:
            observed_hidden.add(scenario.scenario_id)
    if observed_hidden != expected_hidden:
        failures.append(f"unexpected evidence-router blind spot: expected={sorted(expected_hidden)} observed={sorted(observed_hidden)}")
    report = {
        "scenario_count": len(scenarios),
        "hazardous_count": sum(item.hazardous for item in scenarios),
        "profiles": Counter(item.telemetry_profile for item in scenarios),
        "controller_profile_metrics": {
            f"{profile}/{controller}": {
                "n": counts["n"],
                "unsafe_rate": counts["unsafe"] / counts["n"],
                "safe_success": counts["safe_success"] / counts["n"],
                "verification_cost": counts["verification_cost"] / counts["n"],
            }
            for (profile, controller), counts in sorted(grouped.items())
        },
        "expected_observability_blind_spots": len(expected_hidden),
        "failures": failures,
    }
    output = ROOT / "artifacts" / "main" / "simulator_validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, default=dict), encoding="utf-8")
    print(json.dumps(report, indent=2, default=dict))
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
