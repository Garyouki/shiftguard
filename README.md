# ShiftGuard

**Stale does not mean unsafe.** A deterministic simulator and a set of commit-time guards for tool-using LLM agents whose view of infrastructure state can go stale between *read* and *commit*.

[![CI](https://github.com/Garyouki/shiftguard/actions/workflows/ci.yml/badge.svg)](https://github.com/Garyouki/shiftguard/actions/workflows/ci.yml)
[![arXiv](https://img.shields.io/badge/arXiv-2609.29522-b31b1b.svg)](https://arxiv.org/abs/2609.29522)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97-shiftguard--trajectories-yellow)](https://huggingface.co/datasets/youkijiudai/shiftguard-trajectories)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24430268.svg)](https://doi.org/10.5281/zenodo.24430268)

Paper: *Stale Does Not Mean Unsafe: Guard Precision for Tool-Using LLM Agents under Infrastructure State Races*. Zihao Zheng, Jiayu Long, Baichuan Li, Junyi Yao. [arXiv:2609.29522](https://arxiv.org/abs/2609.29522).

## Why

When an agent restarts a job, promotes a snapshot, or changes an ACL, the state it read may have changed by the time it commits. Not every change makes the commit unsafe. ShiftGuard separates three kinds of race:

- **invalidating**: the change breaks a declared safety predicate, so the commit must be blocked.
- **predicate-preserving**: state changed, but the predicate still holds, so blocking is a false alarm.
- **irrelevant**: the change does not touch the action.

It then measures how precisely each runtime guard tells these apart. A coarse guard (for example a global epoch check) is safe but blocks a lot of benign work. A semantic commit predicate can be just as safe at a fraction of the cost.

## What's in here

| Path | Contents |
| --- | --- |
| `src/shiftguard/simulator.py` | 16 infrastructure tasks in 4 domains (scheduler, data pipeline, object store, access control), visible vs. authoritative state, 5 non-atomic fault injectors |
| `src/shiftguard/controllers.py` | Baselines and guards: `always_execute`, fast/strong verification, `escalate`, TTL / confidence / contract / agreement gates, `evidence_router`, `global_epoch_guard`, `read_set_version_guard`, `commit_predicate_guard` plus ablations |
| `src/shiftguard/prompts.py`, `ollama_client.py` | Proposal prompts and a dependency-free Ollama client for collecting new trajectories |
| `scripts/` | Collection, counterfactual replay, analysis, learned router, cost sensitivity, figures |
| `tests/` | 17 deterministic simulator and guard tests |
| `examples/sample_trajectories.jsonl` | 72 frozen proposals from the main study, enough to run the full pipeline offline |

Correctness is scored deterministically. No LLM judge is used anywhere.

## Quick start

The core package and the replay pipeline use only the Python standard library (3.10+).

```bash
git clone https://github.com/Garyouki/shiftguard && cd shiftguard
python -m unittest discover -s tests -v

# Replay the bundled sample under every controller, then summarize
python scripts/replay_controllers.py examples/sample_trajectories.jsonl out/replayed.jsonl
python scripts/analyze_experiment.py out/replayed.jsonl --output-dir out/analysis
cat out/analysis/summary.md
```

## Reproducing the paper

The frozen trajectories (3,456 main-study proposals from Qwen3-4B, Phi-4-mini and Gemma4-8B, plus the Llama-3.2-3B replication, five-seed extension, prompt-caution control, and end-to-end replanning run) are on the Hugging Face Hub:

```bash
pip install huggingface_hub
python scripts/download_data.py            # ~155 MB into ./data

python scripts/replay_controllers.py data/trajectories_v1.jsonl out/main_replayed.jsonl
python scripts/analyze_experiment.py out/main_replayed.jsonl --output-dir out/main_analysis
```

The replay scores both baseline controllers and the guard family. Its output matches all 3,456 rows of `trajectories_v2_guards_replayed.jsonl` exactly.

Other drivers:

```bash
pip install -e ".[analysis]"
python scripts/evaluate_learned_router.py REPLAYED.jsonl --output-dir ROUTER
python scripts/cost_sensitivity.py REPLAYED.jsonl --output COSTS.csv
python scripts/plot_results.py REPLAYED.jsonl --analysis-dir ANALYSIS --output-dir figures
```

### Collecting new trajectories

Collection needs a local [Ollama](https://ollama.com) server. The checkpoint digests used in the paper are recorded in `model_lineage.json` in the dataset.

```bash
python scripts/run_pilot.py --instances-per-template 4 --seeds 13 42 2026
python scripts/run_pilot.py --instances-per-template 4 --seeds 13 42 2026 --models llama3.2:3b
python scripts/run_pilot.py --instances-per-template 4 --seeds 13 42 2026 --prompt-variant aware
```

## Scope

The simulator and the normalized check costs are a controlled research setting. They do not establish production deployment guarantees.

## Citation

```bibtex
@misc{zheng2026staledoesnotmeanunsafe,
  title={Stale Does Not Mean Unsafe: Guard Precision for Tool-Using LLM Agents under Infrastructure State Races},
  author={Zihao Zheng and Jiayu Long and Baichuan Li and Junyi Yao},
  year={2026},
  eprint={2609.29522},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  doi={10.48550/arXiv.2609.29522},
  url={https://arxiv.org/abs/2609.29522}
}
```

## License

Code: MIT (see [LICENSE](LICENSE)). The trajectory dataset has its own card and license on the Hugging Face Hub.
