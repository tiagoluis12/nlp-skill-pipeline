"""Export the non-pilot Irish corpus for final Label Studio annotation."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_pilot_ids(path: Path) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = [payload] if isinstance(payload, dict) else payload
    return {str((row.get("data") or {}).get("source_id")) for row in rows}


def to_labelstudio_task(record: dict) -> dict:
    sentences = record.get("sentences") or []
    text = " ".join(sentences) if sentences else str(record.get("raw_text") or "")
    return {
        "id": f"{record['id']}::irish-gold",
        "data": {
            "text": text,
            "role_type": record["role_type"],
            "country": record.get("country", "IE"),
            "source_id": record["id"],
            "title": record.get("title", ""),
        },
    }


def build_exports(source_jsonl: Path, pilot_json: Path, spot_fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    pilot_ids = load_pilot_ids(pilot_json)
    records = load_jsonl(source_jsonl)
    remaining = [record for record in records if str(record.get("id")) not in pilot_ids]
    if len({record.get("id") for record in remaining}) != len(remaining):
        raise ValueError("Remaining Irish corpus contains duplicate record IDs")
    tasks = [to_labelstudio_task(record) for record in remaining]
    spot_count = max(1, math.ceil(len(tasks) * spot_fraction)) if tasks else 0
    rng = random.Random(seed)
    spot_check = rng.sample(tasks, spot_count)
    return tasks, spot_check


def main() -> None:
    parser = argparse.ArgumentParser(description="Export remaining Irish postings for Label Studio annotation.")
    parser.add_argument("--source-jsonl", type=Path, default=Path("data/clean/irish_final_115.jsonl"))
    parser.add_argument("--pilot-json", type=Path, default=Path("data/annotated/irish_pilot_labelstudio.json"))
    parser.add_argument("--output", type=Path, default=Path("data/annotated/irish_gold_remaining_labelstudio.json"))
    parser.add_argument("--spot-check-output", type=Path, default=Path("data/annotated/irish_gold_spotcheck_labelstudio.json"))
    parser.add_argument("--spot-check-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tasks, spot_check = build_exports(args.source_jsonl, args.pilot_json, args.spot_check_fraction, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(tasks, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.spot_check_output.write_text(json.dumps(spot_check, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "remaining_tasks": len(tasks),
        "spot_check_tasks": len(spot_check),
        "role_counts": dict(sorted(Counter(task["data"]["role_type"] for task in tasks).items())),
        "output": str(args.output),
        "spot_check_output": str(args.spot_check_output),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
