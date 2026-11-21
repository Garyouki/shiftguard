"""Deterministic baseline controllers for counterfactual replay."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .simulator import (
    Outcome,
    Proposal,
    Scenario,
    evidence_supports_execution,
    execute_proposal,
    goal_satisfied,
    state_satisfies,
    successful_state,
)


Controller = Callable[[Scenario, Proposal], Outcome]


@dataclass(frozen=True)
class NamedController:
    name: str
    run: Controller


def always_execute(scenario: Scenario, proposal: Proposal) -> Outcome:
    return execute_proposal(scenario, proposal)


def _verify(scenario: Scenario, proposal: Proposal, *, strong: bool) -> Outcome:
    evidence = scenario.strong_evidence if strong else scenario.fast_evidence
    cost = 4 if strong else 1
    if evidence_supports_execution(scenario, evidence):
        return execute_proposal(scenario, proposal, verification_cost=cost)
    state_success = successful_state(scenario, scenario.authoritative_state)
    reason = "verified_goal_already_satisfied" if state_success else "verification_blocked_action"
    return Outcome(
        scenario.scenario_id,
        False,
        False,
        state_success,
        state_success,
        reason,
        cost,
        0,
        True,
        dict(scenario.authoritative_state),
    )


def always_verify_fast(scenario: Scenario, proposal: Proposal) -> Outcome:
    return _verify(scenario, proposal, strong=False)


def always_verify_strong(scenario: Scenario, proposal: Proposal) -> Outcome:
    return _verify(scenario, proposal, strong=True)


def escalate(scenario: Scenario, proposal: Proposal) -> Outcome:
    state_success = successful_state(scenario, scenario.authoritative_state)
    return Outcome(
        scenario.scenario_id,
        False,
        False,
        state_success,
        state_success,
        "escalated",
        0,
        10,
        False,
        dict(scenario.authoritative_state),
    )


def ttl_gate(scenario: Scenario, proposal: Proposal, threshold: int = 5) -> Outcome:
    if scenario.visible_age > threshold or scenario.last_tool_status != "none":
        return always_verify_strong(scenario, proposal)
    return always_execute(scenario, proposal)


def confidence_gate(scenario: Scenario, proposal: Proposal, threshold: float = 0.8) -> Outcome:
    if proposal.confidence >= threshold:
        return always_execute(scenario, proposal)
    return always_verify_strong(scenario, proposal)


def contract_gate(scenario: Scenario, proposal: Proposal) -> Outcome:
    """Strong precondition/version gate, modeled as an authoritative check."""

    if proposal.expected_version != scenario.strong_evidence.get("version"):
        return always_verify_strong(scenario, proposal)
    return _verify(scenario, proposal, strong=True)


def evidence_router(scenario: Scenario, proposal: Proposal) -> Outcome:
    """Route observable failure signals to the cheapest adequate check.

    The policy deliberately uses only fields exposed to the agent/controller;
    it never reads ``scenario.fault`` or ``scenario.hazardous``.  An unknown
    status takes the conservative strong-verification fallback.
    """

    status = scenario.last_tool_status
    if status == "none":
        if scenario.visible_age <= 5:
            return always_execute(scenario, proposal)
        return always_verify_fast(scenario, proposal)
    if status == "committed_but_not_visible":
        return always_verify_strong(scenario, proposal)
    if status == "timeout_after_dispatch":
        return always_verify_strong(scenario, proposal)
    if status in {"partial_result", "concurrent_update"}:
        return always_verify_fast(scenario, proposal)
    return always_verify_strong(scenario, proposal)


# Backward-compatible function alias for early pilot notebooks/tests.
shift_guard = evidence_router


# --- Commit-time guards -----------------------------------------------------
#
# Guards model an atomic commit hook rather than a pre-commit evidence
# purchase: each performs one authoritative read at commit time (uniform cost
# 1) and differs only in the granularity of the condition it compares.  This
# isolates guard *precision* from evidence *price*, which the verify
# controllers above already vary.

def _guard(scenario: Scenario, proposal: Proposal, *, passes: bool) -> Outcome:
    if passes:
        return execute_proposal(scenario, proposal, verification_cost=1)
    state_success = successful_state(scenario, scenario.authoritative_state)
    reason = "guard_blocked_goal_already_satisfied" if state_success else "guard_blocked_commit"
    return Outcome(
        scenario.scenario_id,
        False,
        False,
        state_success,
        state_success,
        reason,
        1,
        0,
        True,
        dict(scenario.authoritative_state),
    )


def global_epoch_guard(scenario: Scenario, proposal: Proposal) -> Outcome:
    """Coarsest guard: block if anything changed since the agent's read."""

    unchanged = scenario.authoritative_state == scenario.visible_state
    return _guard(scenario, proposal, passes=unchanged)


def read_set_version_guard(scenario: Scenario, proposal: Proposal) -> Outcome:
    """Block if the version of the read resource advanced since the read."""

    unchanged = scenario.authoritative_state.get("version") == scenario.visible_state.get("version")
    return _guard(scenario, proposal, passes=unchanged)


_OPERATIONAL_DROP = "operational"
_DUPLICATE_DROP = "duplicate"


def _predicate_passes(scenario: Scenario, drop: frozenset[str]) -> bool:
    preconditions = dict(scenario.template.preconditions)
    if _OPERATIONAL_DROP in drop:
        operational = next(
            key for key in scenario.template.preconditions if key not in {"policy", "integrity"}
        )
        preconditions.pop(operational, None)
    for clause in drop - {_OPERATIONAL_DROP, _DUPLICATE_DROP}:
        preconditions.pop(clause, None)
    authoritative = scenario.authoritative_state
    if not state_satisfies(authoritative, preconditions):
        return False
    if _DUPLICATE_DROP not in drop and goal_satisfied(scenario, authoritative):
        return False
    return True


def commit_predicate_guard(
    scenario: Scenario, proposal: Proposal, drop: frozenset[str] = frozenset()
) -> Outcome:
    """Re-evaluate the declared safety predicate atomically at commit time.

    ``drop`` removes declared clauses to model contract incompleteness:
    ``policy``, ``integrity``, ``operational`` (the fault-flipped status
    clause), or ``duplicate`` (the not-already-done check).
    """

    return _guard(scenario, proposal, passes=_predicate_passes(scenario, drop))


def _predicate_guard_variant(drop: frozenset[str]) -> Controller:
    def run(scenario: Scenario, proposal: Proposal) -> Outcome:
        return commit_predicate_guard(scenario, proposal, drop)

    return run


PREDICATE_DROP_VARIANTS: tuple[tuple[str, frozenset[str]], ...] = (
    ("predicate_guard_no_policy", frozenset({"policy"})),
    ("predicate_guard_no_integrity", frozenset({"integrity"})),
    ("predicate_guard_no_operational", frozenset({_OPERATIONAL_DROP})),
    ("predicate_guard_no_duplicate", frozenset({_DUPLICATE_DROP})),
)


def baseline_controllers() -> tuple[NamedController, ...]:
    return (
        NamedController("always_execute", always_execute),
        NamedController("always_verify_fast", always_verify_fast),
        NamedController("always_verify_strong", always_verify_strong),
        NamedController("ttl_gate", ttl_gate),
        NamedController("confidence_gate", confidence_gate),
        NamedController("contract_gate", contract_gate),
        NamedController("evidence_router", evidence_router),
        NamedController("escalate", escalate),
        NamedController("global_epoch_guard", global_epoch_guard),
        NamedController("read_set_version_guard", read_set_version_guard),
        NamedController("commit_predicate_guard", commit_predicate_guard),
    ) + tuple(
        NamedController(name, _predicate_guard_variant(drop))
        for name, drop in PREDICATE_DROP_VARIANTS
    )
