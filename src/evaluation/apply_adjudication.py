"""Apply a completed CSV adjudication queue to a normalized BIO export."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.evaluation.build_iaa_disagreement_report import load_tasks


REQUIRED_FIELDS = {"source_id", "token_start", "token_end", "annotator1_labels", "adjudicated_labels", "decision"}


def apply_queue(base_jsonl: Path, queue_csv: Path, output_jsonl: Path) -> dict[str, int]:
    tasks = load_tasks(base_jsonl)
    with queue_csv.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not REQUIRED_FIELDS.issubset(reader.fieldnames or set()):
            missing = sorted(REQUIRED_FIELDS - set(reader.fieldnames or []))
            raise ValueError(f"Adjudication queue is missing columns: {missing}")
        queue_rows = list(reader)

    seen_ranges: set[tuple[str, int, int]] = set()
    applied = 0
    for row in queue_rows:
        source_id = str(row["source_id"])
        if source_id not in tasks:
            raise ValueError(f"Queue contains unknown source_id: {source_id}")
        start = int(row["token_start"])
        end = int(row["token_end"])
        key = (source_id, start, end)
        if key in seen_ranges:
            raise ValueError(f"Duplicate adjudication range: {key}")
        seen_ranges.add(key)
        if start < 0 or end <= start or end > len(tasks[source_id]["labels"]):
            raise ValueError(f"Invalid token range for {source_id}: {start}:{end}")

        expected = row["annotator1_labels"].split()
        final = row["adjudicated_labels"].split()
        current = tasks[source_id]["labels"][start:end]
        if current != expected:
            raise ValueError(f"Annotator 1 labels changed for {source_id}:{start}:{end}")
        if len(final) != end - start:
            raise ValueError(f"Adjudicated label length mismatch for {source_id}:{start}:{end}")
        if not row["decision"].strip() or not row["decision_reason"].strip():
            raise ValueError(f"Incomplete adjudication fields for {source_id}:{start}:{end}")
        tasks[source_id]["labels"][start:end] = final
        applied += 1

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for source_id in sorted(tasks):
            handle.write(json.dumps(tasks[source_id], ensure_ascii=False) + "\n")
    return {"tasks": len(tasks), "adjudication_groups": applied}


def write_conll(jsonl_path: Path, conll_path: Path) -> None:
    conll_path.parent.mkdir(parents=True, exist_ok=True)
    with conll_path.open("w", encoding="utf-8") as output:
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            for token, label in zip(row["tokens"], row["labels"], strict=True):
                output.write(f"{token} {label}\n")
            output.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply a completed adjudication CSV to normalized BIO JSONL.")
    parser.add_argument("--base-jsonl", type=Path, required=True)
    parser.add_argument("--queue-csv", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-conll", type=Path)
    args = parser.parse_args()
    summary = apply_queue(args.base_jsonl, args.queue_csv, args.output_jsonl)
    if args.output_conll is not None:
        write_conll(args.output_jsonl, args.output_conll)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
