from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import requests


def fetch_to_path(url: str, target: Path) -> None:
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    target.write_bytes(response.content)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage ESCO skills resources locally.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/benchmarks/esco"))
    parser.add_argument("--csv-path", type=Path, default=None, help="Existing local skills_en.csv path.")
    parser.add_argument("--csv-url", default=None, help="Direct download URL for skills_en.csv.")
    parser.add_argument("--jsonld-path", type=Path, default=None, help="Existing local JSON-LD path.")
    parser.add_argument("--jsonld-url", default=None, help="Direct download URL for the JSON-LD export.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = args.csv_path
    if args.csv_url:
        csv_path = args.output_dir / "skills_en.csv"
        fetch_to_path(args.csv_url, csv_path)
    if csv_path is None:
        raise SystemExit("Provide --csv-path or --csv-url.")

    skills = pd.read_csv(csv_path)
    print(f"Loaded {len(skills)} ESCO skills rows from {csv_path}")
    print(skills.columns.tolist())

    jsonld_path = args.jsonld_path
    if args.jsonld_url:
        jsonld_path = args.output_dir / "skills_en.jsonld"
        fetch_to_path(args.jsonld_url, jsonld_path)
    if jsonld_path is not None:
        print(f"JSON-LD staged at {jsonld_path}")


if __name__ == "__main__":
    main()
