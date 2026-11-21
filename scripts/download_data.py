"""Download the frozen ShiftGuard trajectories from the Hugging Face Hub.

Usage:
    pip install huggingface_hub
    python scripts/download_data.py            # everything (~155 MB) into ./data
    python scripts/download_data.py --main     # only the main study files
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_ID = "youkijiudai/shiftguard-trajectories"
MAIN = ["trajectories_v1.jsonl", "trajectories_v1_replayed.jsonl", "model_lineage.json", "manifest.json"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--main", action="store_true", help="download only the main study files")
    args = parser.parse_args()
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=args.output_dir,
        allow_patterns=MAIN if args.main else None,
    )
    print(f"Downloaded {REPO_ID} to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
