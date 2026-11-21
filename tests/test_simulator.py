from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shiftguard.controllers import (
    PREDICATE_DROP_VARIANTS,
    always_execute,
    always_verify_fast,
    always_verify_strong,
    commit_predicate_guard,
    evidence_router,
    global_epoch_guard,
    read_set_version_guard,
)
from shiftguard.simulator import Fault, Proposal, generate_scenarios, race_type


class SimulatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.scenarios = generate_scenarios()

    def proposal(self, scenario):
        return Proposal(
            scenario.template.operation,
            scenario.resource_id,
            int(scenario.visible_state["version"]),
            0.9,
        )

    def test_matrix_has_six_faults_for_sixteen_templates(self):
        self.assertEqual(len(self.scenarios), 16 * 6)

    def test_clean_execution_is_safe_and_successful(self):
        for scenario in self.scenarios:
            if scenario.fault is Fault.CLEAN:
                result = always_execute(scenario, self.proposal(scenario))
                self.assertFalse(result.unsafe, scenario.scenario_id)
                self.assertTrue(result.safe_task_success, scenario.scenario_id)

    def test_strong_verification_never_commits_unsafely(self):
        for scenario in self.scenarios:
            result = always_verify_strong(scenario, self.proposal(scenario))
            self.assertFalse(result.unsafe, scenario.scenario_id)

    def test_delayed_visibility_makes_direct_execution_duplicate(self):
        delayed = [item for item in self.scenarios if item.fault is Fault.DELAYED_VISIBILITY]
        self.assertTrue(delayed)
        for scenario in delayed:
            result = always_execute(scenario, self.proposal(scenario))
            self.assertTrue(result.unsafe, scenario.scenario_id)
            self.assertEqual(result.reason, "duplicate_side_effect")

    def test_fast_verification_is_insufficient_for_delayed_visibility(self):
        delayed = [item for item in self.scenarios if item.fault is Fault.DELAYED_VISIBILITY]
        unsafe = [always_verify_fast(item, self.proposal(item)).unsafe for item in delayed]
        self.assertTrue(all(unsafe))

    def test_paired_instances_balance_hazardous_faults(self):
        paired = generate_scenarios(instances_per_template=2)
        for fault in Fault:
            subset = [item for item in paired if item.fault is fault]
            expected = 0 if fault is Fault.CLEAN else len(subset) // 2
            self.assertEqual(sum(item.hazardous for item in subset), expected)

    def test_benign_shift_is_safe_to_execute(self):
        paired = generate_scenarios(instances_per_template=2)
        benign = [item for item in paired if item.fault is not Fault.CLEAN and not item.hazardous]
        self.assertTrue(benign)
        for scenario in benign:
            result = always_execute(scenario, self.proposal(scenario))
            self.assertFalse(result.unsafe, scenario.scenario_id)
            self.assertTrue(result.safe_task_success, scenario.scenario_id)

    def test_evidence_router_is_safe_and_cheaper_than_strong_verification(self):
        paired = generate_scenarios(instances_per_template=2)
        results = [evidence_router(item, self.proposal(item)) for item in paired]
        self.assertFalse(any(result.unsafe for result in results))
        mean_cost = sum(result.verification_cost for result in results) / len(results)
        self.assertLess(mean_cost, 4.0)

    def test_degraded_telemetry_breaks_one_to_one_fault_labels(self):
        stress = generate_scenarios(instances_per_template=4)
        degraded = [item for item in stress if item.telemetry_profile == "degraded"]
        self.assertTrue(degraded)
        statuses = {item.last_tool_status for item in degraded}
        self.assertIn("transport_error_unknown", statuses)
        self.assertIn("success_with_warning", statuses)
        hidden_concurrency = [
            item for item in degraded
            if item.fault is Fault.CONCURRENT_MUTATION and item.hazardous
        ]
        self.assertTrue(hidden_concurrency)
        self.assertTrue(all(item.last_tool_status == "none" for item in hidden_concurrency))


class GuardPrecisionTests(unittest.TestCase):
    """Race taxonomy and commit-time guard granularity properties."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.scenarios = generate_scenarios(instances_per_template=4)

    def proposal(self, scenario):
        return Proposal(
            scenario.template.operation,
            scenario.resource_id,
            int(scenario.visible_state["version"]),
            0.9,
        )

    def benign_block(self, scenario, outcome) -> bool:
        counterfactual = always_execute(scenario, self.proposal(scenario))
        return (
            not outcome.executed
            and counterfactual.executed
            and not counterfactual.unsafe
            and counterfactual.safe_task_success
        )

    def test_race_type_partitions_the_grid(self):
        for scenario in self.scenarios:
            label = race_type(scenario)
            if scenario.fault is Fault.CLEAN:
                self.assertEqual(label, "none", scenario.scenario_id)
            elif scenario.hazardous:
                self.assertEqual(label, "invalidating", scenario.scenario_id)
            elif scenario.fault is Fault.AMBIGUOUS_COMMIT:
                self.assertEqual(label, "irrelevant", scenario.scenario_id)
            else:
                self.assertEqual(label, "predicate_preserving", scenario.scenario_id)

    def test_all_guards_prevent_unsafe_commits(self):
        for guard in (global_epoch_guard, read_set_version_guard, commit_predicate_guard):
            for scenario in self.scenarios:
                outcome = guard(scenario, self.proposal(scenario))
                self.assertFalse(outcome.unsafe, f"{guard.__name__} {scenario.scenario_id}")
                self.assertEqual(outcome.verification_cost, 1)

    def test_global_epoch_guard_blocks_every_state_change(self):
        for scenario in self.scenarios:
            outcome = global_epoch_guard(scenario, self.proposal(scenario))
            if scenario.fault is Fault.CLEAN:
                self.assertTrue(outcome.executed, scenario.scenario_id)
            else:
                self.assertFalse(outcome.executed, scenario.scenario_id)

    def test_version_guard_blocks_predicate_preserving_but_passes_irrelevant(self):
        for scenario in self.scenarios:
            outcome = read_set_version_guard(scenario, self.proposal(scenario))
            label = race_type(scenario)
            if label in {"none", "irrelevant"}:
                self.assertTrue(outcome.executed, scenario.scenario_id)
            elif label == "predicate_preserving":
                self.assertTrue(self.benign_block(scenario, outcome), scenario.scenario_id)

    def test_predicate_guard_has_no_benign_interventions(self):
        for scenario in self.scenarios:
            outcome = commit_predicate_guard(scenario, self.proposal(scenario))
            self.assertFalse(self.benign_block(scenario, outcome), scenario.scenario_id)
            if race_type(scenario) == "invalidating":
                self.assertFalse(outcome.executed, scenario.scenario_id)

    def test_dropping_policy_clause_leaks_exactly_concurrent_hazards(self):
        drop = dict(PREDICATE_DROP_VARIANTS)["predicate_guard_no_policy"]
        leaked = {
            scenario.scenario_id
            for scenario in self.scenarios
            if commit_predicate_guard(scenario, self.proposal(scenario), drop).unsafe
        }
        expected = {
            scenario.scenario_id
            for scenario in self.scenarios
            if scenario.fault is Fault.CONCURRENT_MUTATION and scenario.hazardous
        }
        self.assertEqual(leaked, expected)

    def test_dropping_duplicate_clause_leaks_only_completed_effect_hazards(self):
        drop = dict(PREDICATE_DROP_VARIANTS)["predicate_guard_no_duplicate"]
        leaked = [
            scenario
            for scenario in self.scenarios
            if commit_predicate_guard(scenario, self.proposal(scenario), drop).unsafe
        ]
        self.assertTrue(leaked)
        for scenario in leaked:
            self.assertTrue(scenario.hazardous, scenario.scenario_id)
            self.assertIn(
                scenario.fault,
                {Fault.DELAYED_VISIBILITY, Fault.AMBIGUOUS_COMMIT, Fault.PARTIAL_SUCCESS},
                scenario.scenario_id,
            )

    def test_dropping_operational_clause_leaks_only_stale_hazards(self):
        drop = dict(PREDICATE_DROP_VARIANTS)["predicate_guard_no_operational"]
        leaked = [
            scenario
            for scenario in self.scenarios
            if commit_predicate_guard(scenario, self.proposal(scenario), drop).unsafe
        ]
        self.assertTrue(leaked)
        for scenario in leaked:
            self.assertTrue(scenario.hazardous, scenario.scenario_id)
            self.assertIs(scenario.fault, Fault.STALE_READ, scenario.scenario_id)


if __name__ == "__main__":
    unittest.main()
