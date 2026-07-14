from __future__ import annotations

import argparse
from pathlib import Path

import requests


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the Green et al. benchmark ZIP and stage it locally.")
    parser.add_argument("--zip-url", required=True, help="Direct download URL for the supplementary ZIP.")
    parser.add_argument("--output", type=Path, default=Path("data/benchmarks/green_et_al"))
    parser.add_argument("--filename", default="green_et_al.zip")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / args.filename
    response = requests.get(args.zip_url, timeout=60)
    response.raise_for_status()
    target.write_bytes(response.content)
    print(f"Saved {target}")


if __name__ == "__main__":
    main()
