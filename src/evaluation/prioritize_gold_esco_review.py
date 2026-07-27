from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _float(value: str | None) -> float:
    try:
        return float(value) if value not in (None, "") else 0.0
    except ValueError:
        return 0.0


def prioritize_review_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    prioritized: list[dict[str, str]] = []
    for row in rows:
        lexicon_uri = (row.get("lexicon_uri") or "").strip()
        candidate_1_uri = (row.get("candidate_1_uri") or "").strip()
        candidate_2_score = _float(row.get("candidate_2_score"))
        candidate_1_score = _float(row.get("candidate_1_score"))
        gap = candidate_1_score - candidate_2_score

        if lexicon_uri:
            provisional_uri = lexicon_uri
            provisional_source = "lexicon_exact"
        elif candidate_1_uri:
            provisional_uri = candidate_1_uri
            provisional_source = "embedding_top1"
        else:
            provisional_uri = ""
            provisional_source = "needs_review"

        review_priority = (
            (2.0 if not lexicon_uri else 0.0)
            + max(0.0, 1.0 - candidate_1_score)
            + max(0.0, 0.2 - gap)
        )

        enriched = dict(row)
        enriched["provisional_gold_uri"] = provisional_uri
        enriched["provisional_source"] = provisional_source
        enriched["review_priority"] = f"{review_priority:.6f}"
        enriched["confidence_gap"] = f"{gap:.6f}"
        enriched["needs_review"] = "yes" if not lexicon_uri or gap < 0.2 else "no"
        prioritized.append(enriched)

    prioritized.sort(
        key=lambda row: (
            -_float(row.get("review_priority")),
            row.get("role_type", ""),
            int(float(row.get("posting_id") or 0)),
            int(float(row.get("span_index") or 0)),
        )
    )
    return prioritized


def main() -> None:
    parser = argparse.ArgumentParser(description="Prioritize the gold ESCO review file.")
    parser.add_argument("--input", type=Path, default=Path("results/gold_esco_review.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/gold_esco_review_priority.csv"))
    parser.add_argument("--summary-output", type=Path, default=Path("results/gold_esco_review_priority.json"))
    args = parser.parse_args()

    with args.input.open("r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    prioritized = prioritize_review_rows(rows)
    fieldnames = sorted({key for row in prioritized for key in row.keys()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(prioritized)

    summary = {
        "input_rows": len(rows),
        "output_rows": len(prioritized),
        "needs_review": sum(1 for row in prioritized if row.get("needs_review") == "yes"),
        "lexicon_exact": sum(1 for row in prioritized if row.get("provisional_source") == "lexicon_exact"),
        "embedding_top1": sum(1 for row in prioritized if row.get("provisional_source") == "embedding_top1"),
    }
    args.summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
