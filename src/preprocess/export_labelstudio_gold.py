from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


def load_records(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                records.append(json.loads(line))
    return records


def build_text(record: dict) -> str:
    sentences = record.get("sentences") or []
    if sentences:
        return " ".join(sentences)
    return record.get("raw_text") or ""


def load_excluded_ids(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        raise TypeError(f"Unsupported exclusion file format in {path}")

    excluded: set[str] = set()
    for row in payload:
        source_id = (row.get("data") or {}).get("source_id") or row.get("source_id")
        if source_id is not None:
            excluded.add(str(source_id))
    return excluded


def select_stratified_records(clean_dir: Path, per_role: int, excluded_ids: set[str], seed: int) -> list[dict]:
    rng = random.Random(seed)
    grouped: list[tuple[str, list[dict]]] = []

    for role_path in sorted(clean_dir.glob("*.jsonl")):
        if role_path.name == ".gitkeep":
            continue
        role_records = [
            record
            for record in load_records(role_path)
            if str(record.get("id")) not in excluded_ids
        ]
        rng.shuffle(role_records)
        grouped.append((role_path.stem, role_records))

    selected: list[dict] = []
    for index in range(per_role):
        for role_type, role_records in grouped:
            if index >= len(role_records):
                continue
            record = role_records[index]
            selected.append(
                {
                    "id": f"{record.get('id')}::{index + 1}",
                    "data": {
                        "text": build_text(record),
                        "role_type": role_type,
                        "country": record.get("country"),
                        "source_id": record.get("id"),
                    },
                }
            )
    return selected


def make_spot_check(records: list[dict], fraction: float, seed: int) -> list[dict]:
    if not records:
        return []
    count = max(1, math.ceil(len(records) * fraction))
    rng = random.Random(seed)
    subset = records[:]
    rng.shuffle(subset)
    return subset[:count]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a stratified Label Studio gold set.")
    parser.add_argument("--clean-dir", type=Path, default=Path("data/clean"))
    parser.add_argument("--pilot-json", type=Path, default=Path("data/annotated/pilot_labelstudio.json"))
    parser.add_argument("--output", type=Path, default=Path("data/annotated/gold_labelstudio.json"))
    parser.add_argument("--spot-check-output", type=Path, default=Path("data/annotated/gold_spotcheck_labelstudio.json"))
    parser.add_argument("--per-role", type=int, default=25)
    parser.add_argument("--spot-check-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    excluded_ids = load_excluded_ids(args.pilot_json)
    gold = select_stratified_records(args.clean_dir, args.per_role, excluded_ids, args.seed)
    spot_check = make_spot_check(gold, args.spot_check_fraction, args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(gold, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.spot_check_output.write_text(json.dumps(spot_check, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summary = {
        "gold_records": len(gold),
        "spot_check_records": len(spot_check),
        "output": str(args.output),
        "spot_check_output": str(args.spot_check_output),
        "excluded_ids": len(excluded_ids),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
