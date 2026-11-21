"""Run resumable local-model proposal and counterfactual baseline experiments."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shiftguard.controllers import baseline_controllers
from shiftguard.ollama_client import OllamaClient, parse_json_content
from shiftguard.prompts import action_schema, proposal_messages
from shiftguard.simulator import Domain, generate_scenarios, proposal_from_mapping


DEFAULT_MODELS = ("qwen3:4b", "phi4-mini:latest", "gemma4:latest")
DEFAULT_SEEDS = (13, 42, 2026)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--instances-per-template", type=int, default=1)
    parser.add_argument("--templates-per-domain", type=int, default=0, help="0 means all")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--prompt-variant", choices=["base", "aware"], default="base")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "pilot" / "trajectories.jsonl")
    return parser.parse_args()


def select_scenarios(args: argparse.Namespace):
    scenarios = list(generate_scenarios(args.instances_per_template))
    if args.templates_per_domain:
        allowed: set[tuple[Domain, str]] = set()
        for domain in Domain:
            names = []
            for item in scenarios:
                key = (item.template.domain, item.template.task_name)
                if item.template.domain is domain and key not in allowed and item.template.task_name not in names:
                    names.append(item.template.task_name)
                if len(names) >= args.templates_per_domain:
                    break
            allowed.update((domain, name) for name in names)
        scenarios = [item for item in scenarios if (item.template.domain, item.template.task_name) in allowed]
    if args.limit:
        scenarios = scenarios[: args.limit]
    return scenarios


def load_completed(path: Path) -> set[tuple[str, str, int]]:
    completed: set[tuple[str, str, int]] = set()
    if not path.exists():
        return completed
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        completed.add((row["model"], row["scenario_id"], int(row["seed"])))
    return completed


def outcome_dict(outcome) -> dict[str, Any]:
    result = asdict(outcome)
    result.pop("final_state", None)
    return result


def main() -> int:
    args = parse_args()
    scenarios = select_scenarios(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = load_completed(args.output)
    client = OllamaClient(timeout=300)
    total = len(args.models) * len(args.seeds) * len(scenarios)
    finished = len(completed)
    print(f"planned={total} already_completed={finished} scenarios={len(scenarios)}", flush=True)

    with args.output.open("a", encoding="utf-8") as handle:
        for model in args.models:
            for scenario in scenarios:
                for seed in args.seeds:
                    key = (model, scenario.scenario_id, seed)
                    if key in completed:
                        continue
                    started = time.perf_counter()
                    error = None
                    reply = None
                    try:
                        reply = client.chat(
                            model=model,
                            messages=proposal_messages(scenario, args.prompt_variant),
                            format_schema=action_schema(scenario),
                            seed=seed,
                            temperature=args.temperature,
                            top_p=0.9,
                            num_ctx=4096,
                            num_predict=128,
                            think=False,
                        )
                        parsed = parse_json_content(reply)
                    except Exception as exc:  # preserve failed episodes as artifacts
                        parsed = None
                        error = f"{type(exc).__name__}: {exc}"
                    proposal = proposal_from_mapping(parsed)
                    controller_results = {
                        controller.name: outcome_dict(controller.run(scenario, proposal))
                        for controller in baseline_controllers()
                    }
                    row = {
                        "model": model,
                        "seed": seed,
                        "prompt_variant": args.prompt_variant,
                        "scenario_id": scenario.scenario_id,
                        "domain": scenario.template.domain.value,
                        "task_name": scenario.template.task_name,
                        "fault": scenario.fault.value,
                        "operation": scenario.template.operation,
                        "blast_radius": scenario.template.blast_radius,
                        "visible_age": scenario.visible_age,
                        "last_tool_status": scenario.last_tool_status,
                        "telemetry_profile": scenario.telemetry_profile,
                        "hazardous": scenario.hazardous,
                        "visible_version": scenario.visible_state["version"],
                        "authoritative_version": scenario.authoritative_state["version"],
                        "proposal": asdict(proposal),
                        "raw_content": reply.content if reply else "",
                        "prompt_tokens": reply.prompt_tokens if reply else 0,
                        "output_tokens": reply.output_tokens if reply else 0,
                        "model_seconds": (reply.total_duration_ns / 1e9) if reply else 0,
                        "wall_seconds": time.perf_counter() - started,
                        "done_reason": reply.done_reason if reply else None,
                        "error": error,
                        "controllers": controller_results,
                    }
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                    finished += 1
                    print(
                        f"[{finished}/{total}] {model} {scenario.scenario_id} seed={seed} "
                        f"action={proposal.action} conf={proposal.confidence:.2f} "
                        f"wall={row['wall_seconds']:.2f}s",
                        flush=True,
                    )
            try:
                client.unload(model)
            except Exception as exc:
                print(f"warning: failed to unload {model}: {exc}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
