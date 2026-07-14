from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_records(path: Path) -> list[dict]:
    records = []
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


def select_pilot_records(clean_dir: Path, per_role: int) -> list[dict]:
    grouped: list[tuple[str, list[dict]]] = []
    for role_path in sorted(clean_dir.glob("*.jsonl")):
        if role_path.name == ".gitkeep":
            continue
        role_records = load_records(role_path)[:per_role]
        grouped.append((role_path.stem, role_records))

    pilot = []
    for index in range(per_role):
        for _, role_records in grouped:
            if index >= len(role_records):
                continue
            record = role_records[index]
            pilot.append(
                {
                    "id": f"{record.get('id')}::{index + 1}",
                    "data": {
                        "text": build_text(record),
                        "role_type": record.get("role_type"),
                        "country": record.get("country"),
                        "source_id": record.get("id"),
                    },
                }
            )
    return pilot


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a stratified Label Studio pilot set.")
    parser.add_argument("--clean-dir", type=Path, default=Path("data/clean"))
    parser.add_argument("--output", type=Path, default=Path("data/annotated/pilot_labelstudio.json"))
    parser.add_argument("--per-role", type=int, default=6)
    args = parser.parse_args()

    pilot = select_pilot_records(args.clean_dir, args.per_role)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(pilot, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"records": len(pilot), "output": str(args.output)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
