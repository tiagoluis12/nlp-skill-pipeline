from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

from src.esco.ranker import load_span_ranker, rank_candidate_texts
from src.extraction.lexicon_baseline import build_lexicon_index, load_esco_skills, match_span_detail


def _bio_to_spans(labels: Sequence[str]) -> list[dict[str, int | str]]:
    spans: list[dict[str, int | str]] = []
    start = None
    current = None
    for idx, label in enumerate(list(labels) + ["O"]):
        if label.startswith("B-"):
            if start is not None and current is not None:
                spans.append({"start": start, "end": idx, "label": current})
            start = idx
            current = label[2:]
        elif label.startswith("I-") and current == label[2:]:
            continue
        else:
            if start is not None and current is not None:
                spans.append({"start": start, "end": idx, "label": current})
            start = None
            current = None
    return spans


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _span_text(tokens: Sequence[str], start: int, end: int) -> str:
    return " ".join(tokens[start:end])


def _context_text(tokens: Sequence[str], start: int, end: int, window: int = 8) -> str:
    left = " ".join(tokens[max(0, start - window) : start])
    span = " ".join(tokens[start:end])
    right = " ".join(tokens[end : min(len(tokens), end + window)])
    return " ".join(part for part in [left, f"<<{span}>>", right] if part).strip()


def build_gold_esco_review(
    gold_jsonl: Path,
    esco_csv: Path,
    embeddings_path: Path,
    index_path: Path,
    labels_path: Path,
    top_k: int = 5,
) -> list[dict[str, object]]:
    skills = load_esco_skills(esco_csv)
    lexicon_index = build_lexicon_index(skills)
    ranker = load_span_ranker(embeddings_path, index_path, labels_path)

    rows: list[dict[str, object]] = []
    for record in _load_jsonl(gold_jsonl):
        gold_spans = _bio_to_spans(record.get("labels") or [])
        for span_idx, span in enumerate(gold_spans, start=1):
            text = _span_text(record["tokens"], int(span["start"]), int(span["end"]))
            lexicon_match = match_span_detail(text, lexicon_index)
            scores, indices = rank_candidate_texts([text], ranker, top_k=top_k)
            candidates = []
            for rank, (score, idx) in enumerate(zip(scores[0], indices[0], strict=True), start=1):
                if idx < 0:
                    continue
                candidates.append(
                    {
                        "rank": rank,
                        "concept_uri": ranker.artifacts.concept_uris[idx],
                        "label": ranker.artifacts.labels[idx],
                        "score": float(score),
                    }
                )
            row = {
                "posting_id": record.get("id"),
                "source_id": record.get("source_id"),
                "role_type": record.get("role_type"),
                "span_index": span_idx,
                "span_start": span["start"],
                "span_end": span["end"],
                "span_label": span["label"],
                "span_text": text,
                "context": _context_text(record["tokens"], int(span["start"]), int(span["end"])),
                "lexicon_uri": lexicon_match.concept_uri or "",
                "lexicon_label": lexicon_match.matched_label or "",
                "lexicon_score": lexicon_match.score,
                "gold_concept_uri": "",
                "reviewer_notes": "",
            }
            for candidate in candidates:
                rank = int(candidate["rank"])
                row[f"candidate_{rank}_uri"] = candidate["concept_uri"]
                row[f"candidate_{rank}_label"] = candidate["label"]
                row[f"candidate_{rank}_score"] = candidate["score"]
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Export gold spans for ESCO concept-linking review.")
    parser.add_argument("--gold-jsonl", type=Path, default=Path("data/annotated/gold_annotator1.normalized.jsonl"))
    parser.add_argument("--esco-csv", type=Path, default=Path("data/benchmarks/esco/skills_en.csv"))
    parser.add_argument("--embeddings", type=Path, default=Path("models/esco_embeddings.npy"))
    parser.add_argument("--index", type=Path, default=Path("models/esco_faiss.index"))
    parser.add_argument("--labels", type=Path, default=Path("models/esco_embedding_labels.json"))
    parser.add_argument("--output", type=Path, default=Path("results/gold_esco_review.csv"))
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    rows = build_gold_esco_review(args.gold_jsonl, args.esco_csv, args.embeddings, args.index, args.labels, top_k=args.top_k)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"output": str(args.output), "rows": len(rows)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
