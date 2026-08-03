"""Redact contact details from collected JSONL records before release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.preprocess.clean import redact_pii


def scrub_file(input_path: Path, output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with input_path.open(encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as target:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            for field in ("description", "raw_text"):
                if isinstance(record.get(field), str):
                    record[field] = redact_pii(record[field])
            target.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Redact emails and phone numbers from JSONL records.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    total = 0
    for input_path in sorted(args.input_dir.glob("*.jsonl")):
        total += scrub_file(input_path, args.output_dir / input_path.name)
    print(json.dumps({"records_scrubbed": total, "output_dir": str(args.output_dir)}))


if __name__ == "__main__":
    main()
