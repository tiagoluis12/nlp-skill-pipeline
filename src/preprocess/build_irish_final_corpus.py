"""Materialize the accepted Irish sample and export its annotation pilot."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from src.preprocess.clean import load_sentence_nlp
from src.preprocess.screen_irish_corpus import IRISH_TARGET_ROLE_COUNTS

ROLE_ORDER = tuple(IRISH_TARGET_ROLE_COUNTS)


def load_candidates(path: Path) -> dict[str, dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                candidates[str(record["id"])] = record
    return candidates


def load_accepted_ids(queue_dir: Path) -> dict[str, list[str]]:
    accepted: dict[str, list[str]] = {}
    for role_type in ROLE_ORDER:
        path = queue_dir / f"{role_type}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        role_ids = [row["id"] for row in rows if row.get("manual_decision", "").strip().lower() == "accept"]
        accepted[role_type] = role_ids
    return accepted


def materialize_records(
    accepted_ids: dict[str, list[str]],
    candidates: dict[str, dict[str, Any]],
    nlp,
    *,
    target_counts: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    target_counts = target_counts or IRISH_TARGET_ROLE_COUNTS
    selected_ids: set[str] = set()
    output: list[dict[str, Any]] = []
    for role_type in ROLE_ORDER:
        role_ids = accepted_ids.get(role_type, [])
        expected = target_counts[role_type]
        if len(role_ids) != expected:
            raise ValueError(f"{role_type}: expected {expected} accepted records, found {len(role_ids)}")
        for source_id in role_ids:
            if source_id in selected_ids:
                raise ValueError(f"Duplicate accepted posting ID: {source_id}")
            selected_ids.add(source_id)
            source = candidates.get(source_id)
            if source is None:
                raise ValueError(f"Accepted ID not found in candidate corpus: {source_id}")
            if source.get("country") != "IE":
                raise ValueError(f"Accepted posting is not Irish: {source_id}")
            if source.get("screening_decision") == "excluded":
                raise ValueError(f"Accepted posting was screened out: {source_id}")
            raw_text = source.get("raw_text", "").strip()
            if not raw_text:
                raise ValueError(f"Accepted posting has no description: {source_id}")
            sentences = [sentence.text.strip() for sentence in nlp(raw_text).sents if sentence.text.strip()]
            output.append(
                {
                    "id": source_id,
                    "role_type": role_type,
                    "country": "IE",
                    "title": source.get("title"),
                    "company": source.get("company"),
                    "location": source.get("location"),
                    "published_at": source.get("published_at"),
                    "source": source.get("source"),
                    "source_file": source.get("source_file"),
                    "url": source.get("url"),
                    "sentences": sentences,
                    "raw_text": raw_text,
                }
            )
    return output


def build_pilot_tasks(records: Iterable[dict[str, Any]], *, per_role: int = 6) -> list[dict[str, Any]]:
    grouped = {role_type: [] for role_type in ROLE_ORDER}
    for record in records:
        if record.get("role_type") in grouped:
            grouped[record["role_type"]].append(record)
    tasks: list[dict[str, Any]] = []
    for role_type in ROLE_ORDER:
        if len(grouped[role_type]) < per_role:
            raise ValueError(f"{role_type}: need {per_role} pilot records, found {len(grouped[role_type])}")
        for index, record in enumerate(grouped[role_type][:per_role], start=1):
            tasks.append(
                {
                    "id": f"irish-pilot-{len(tasks) + 1:03d}",
                    "data": {
                        "text": " ".join(record["sentences"]),
                        "role_type": role_type,
                        "country": "IE",
                        "source_id": record["id"],
                        "title": record.get("title"),
                        "company": record.get("company"),
                    },
                }
            )
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the accepted Irish corpus and its Label Studio pilot.")
    parser.add_argument("--queue-dir", type=Path, default=Path("results/irish_review_queues"))
    parser.add_argument("--candidates", type=Path, default=Path("data/clean/irish_screened_candidates.jsonl"))
    parser.add_argument("--corpus-output", type=Path, default=Path("data/clean/irish_final_115.jsonl"))
    parser.add_argument("--pilot-output", type=Path, default=Path("data/annotated/irish_pilot_labelstudio.json"))
    parser.add_argument("--summary-output", type=Path, default=Path("results/irish_final_corpus_summary.json"))
    parser.add_argument("--pilot-per-role", type=int, default=6)
    args = parser.parse_args()

    accepted_ids = load_accepted_ids(args.queue_dir)
    candidates = load_candidates(args.candidates)
    nlp = load_sentence_nlp()
    records = materialize_records(accepted_ids, candidates, nlp)
    pilot = build_pilot_tasks(records, per_role=args.pilot_per_role)

    args.corpus_output.parent.mkdir(parents=True, exist_ok=True)
    with args.corpus_output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    args.pilot_output.parent.mkdir(parents=True, exist_ok=True)
    args.pilot_output.write_text(json.dumps(pilot, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    summary = {
        "corpus_records": len(records),
        "corpus_role_counts": dict(Counter(record["role_type"] for record in records)),
        "pilot_records": len(pilot),
        "pilot_role_counts": dict(Counter(task["data"]["role_type"] for task in pilot)),
        "corpus_output": str(args.corpus_output),
        "pilot_output": str(args.pilot_output),
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
