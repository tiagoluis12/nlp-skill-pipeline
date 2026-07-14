from __future__ import annotations

import argparse
import json
from pathlib import Path


def count_jsonl_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def mean_sentence_count(path: Path) -> float:
    if not path.exists():
        return 0.0
    total = 0
    rows = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            record = json.loads(line)
            total += len(record.get("sentences", []))
            rows += 1
    return total / rows if rows else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise raw and cleaned corpus sizes.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--clean-dir", type=Path, default=Path("data/clean"))
    parser.add_argument("--output", type=Path, default=Path("results/corpus_stats.json"))
    args = parser.parse_args()

    roles = sorted(
        path.stem
        for path in args.raw_dir.glob("*.jsonl")
        if path.name != ".gitkeep"
    )
    stats = {}
    for role in roles:
        raw_path = args.raw_dir / f"{role}.jsonl"
        clean_path = args.clean_dir / f"{role}.jsonl"
        raw_rows = count_jsonl_lines(raw_path)
        clean_rows = count_jsonl_lines(clean_path)
        stats[role] = {
            "raw_rows": raw_rows,
            "clean_rows": clean_rows,
            "dedup_removed": max(raw_rows - clean_rows, 0),
            "dedup_rate": round((raw_rows - clean_rows) / raw_rows, 4) if raw_rows else 0.0,
            "avg_sentences_clean": round(mean_sentence_count(clean_path), 2),
        }

    overall_raw = sum(item["raw_rows"] for item in stats.values())
    overall_clean = sum(item["clean_rows"] for item in stats.values())
    overall = {
        "raw_rows": overall_raw,
        "clean_rows": overall_clean,
        "dedup_removed": overall_raw - overall_clean,
        "dedup_rate": round((overall_raw - overall_clean) / overall_raw, 4) if overall_raw else 0.0,
        "avg_sentences_clean": round(
            sum(mean_sentence_count(args.clean_dir / f"{role}.jsonl") for role in roles) / len(roles), 2
        )
        if roles
        else 0.0,
    }

    payload = {"roles": stats, "overall": overall}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
