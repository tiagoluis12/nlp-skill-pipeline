from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from src.esco.ranker import load_lexicon_index, load_span_ranker, rank_span_candidates


DEFAULT_REVIEW_CSV = Path("results/gold_esco_review_priority.csv")
DEFAULT_NER_REPORT = Path("results/ner_gold.json")
DEFAULT_OUTPUT_DIR = Path("results/audit_traces")
DEFAULT_ESCO_CSV = Path("data/benchmarks/esco/skills_en.csv")
DEFAULT_EMBEDDINGS = Path("models/esco_embeddings.npy")
DEFAULT_INDEX = Path("models/esco_faiss.index")
DEFAULT_LABELS = Path("models/esco_embedding_labels.json")


def _load_review_rows(review_csv: Path) -> list[dict[str, object]]:
    with review_csv.open("r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return [row for row in rows if (row.get("gold_concept_uri") or "").strip()]


def _posting_key(value: object) -> str:
    return str(value)


def _int_value(value: object) -> int:
    return int(float(value))


def _source_sentence(tokens: Sequence[object], start: int, end: int) -> str:
    sentence_end_tokens = {".", "!", "?"}
    left = start
    while left > 0 and str(tokens[left - 1]) not in sentence_end_tokens:
        left -= 1

    right = end
    while right < len(tokens) and str(tokens[right]) not in sentence_end_tokens:
        right += 1
    if right < len(tokens):
        right += 1

    return " ".join(str(token) for token in tokens[left:right]).strip()


def _group_review_rows(rows: Iterable[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[_posting_key(row.get("posting_id"))].append(row)
    for row_list in grouped.values():
        row_list.sort(key=lambda row: (_int_value(row.get("span_start", 0)), _int_value(row.get("span_end", 0))))
    return grouped


def _overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return min(a_end, b_end) - max(a_start, b_start)


def _match_gold_span(
    predicted: dict[str, object],
    gold_rows: Sequence[dict[str, object]],
) -> tuple[int | None, dict[str, object] | None, str]:
    pred_start = _int_value(predicted["start"])
    pred_end = _int_value(predicted["end"])
    pred_label = str(predicted["label"])

    for index, row in enumerate(gold_rows):
        if (
            _int_value(row.get("span_start", 0)) == pred_start
            and _int_value(row.get("span_end", 0)) == pred_end
            and str(row.get("span_label", "")) == pred_label
        ):
            return index, row, "exact"

    best_row: dict[str, object] | None = None
    best_index: int | None = None
    best_overlap = 0
    for index, row in enumerate(gold_rows):
        if str(row.get("span_label", "")) != pred_label:
            continue
        overlap = _overlap(
            pred_start,
            pred_end,
            _int_value(row.get("span_start", 0)),
            _int_value(row.get("span_end", 0)),
        )
        if overlap > best_overlap:
            best_overlap = overlap
            best_row = row
            best_index = index
    if best_row is not None and best_overlap > 0:
        return best_index, best_row, "boundary"
    return None, None, "none"


def _best_sentence(record: dict[str, object], span: dict[str, object]) -> str:
    sentence = str(span.get("source_sentence") or "").strip()
    if sentence:
        return sentence
    tokens = record.get("tokens")
    if isinstance(tokens, Sequence):
        try:
            start = _int_value(span.get("start", span.get("span_start", 0)))
            end = _int_value(span.get("end", span.get("span_end", 0)))
            return _source_sentence(tokens, start, end)
        except Exception:
            pass
    return str(record.get("source_text") or "").strip()


def _categorize_predicted_span(
    predicted_uri: str | None,
    gold_row: dict[str, object] | None,
    match_type: str,
    top5_candidates: Sequence[dict[str, object]],
) -> str:
    if gold_row is None:
        return "false_positive"

    gold_uri = str(gold_row.get("gold_concept_uri") or "").strip()
    gold_label = str(gold_row.get("span_label") or "")

    if match_type == "boundary":
        return "boundary_error"
    if predicted_uri == gold_uri and gold_uri:
        return "correct"
    if not gold_uri or not any(str(candidate.get("concept_uri")) == gold_uri for candidate in top5_candidates):
        return "esco_coverage_gap"
    if gold_label == "SKILL_TRANS":
        return "soft_skill_ambiguity"
    return "esco_linking_error"


def build_audit_trace_records(
    documents: Sequence[dict[str, object]],
    gold_rows: Sequence[dict[str, object]],
    ranker,
    lexicon_index: dict[str, str],
    top_k: int = 5,
) -> dict[str, list[dict[str, object]]]:
    gold_by_posting = _group_review_rows(gold_rows)
    traces: dict[str, list[dict[str, object]]] = {}

    for record in documents:
        posting_id = _posting_key(record.get("id"))
        posting_gold_rows = list(gold_by_posting.get(posting_id, []))
        unmatched_gold = set(range(len(posting_gold_rows)))
        posting_traces: list[dict[str, object]] = []

        predicted_spans = list(record.get("predicted_spans") or [])
        predicted_spans.sort(key=lambda span: (_int_value(span.get("start", 0)), _int_value(span.get("end", 0))))

        for span in predicted_spans:
            candidates = rank_span_candidates(
                str(span.get("text") or ""),
                ranker,
                lexicon_index=lexicon_index,
                top_k=top_k,
            )
            predicted_uri = str(candidates[0]["concept_uri"]) if candidates else None
            gold_index, gold_row, match_type = _match_gold_span(span, posting_gold_rows)

            if gold_row is not None:
                unmatched_gold.discard(gold_index)
                gold_uri = str(gold_row.get("gold_concept_uri") or "").strip()
            else:
                gold_uri = ""

            posting_traces.append(
                {
                    "record_type": "predicted_span",
                    "posting_id": posting_id,
                    "source_id": record.get("source_id"),
                    "role_type": record.get("source"),
                    "sentence": _best_sentence(record, span),
                    "span_start": span.get("start"),
                    "span_end": span.get("end"),
                    "span": span.get("text"),
                    "span_confidence": span.get("confidence"),
                    "ner_label": span.get("label"),
                    "top5_candidates": candidates,
                    "predicted_uri": predicted_uri,
                    "gold_uri": gold_uri,
                    "gold_label": gold_row.get("span_label") if gold_row else None,
                    "gold_span_start": gold_row.get("span_start") if gold_row else None,
                    "gold_span_end": gold_row.get("span_end") if gold_row else None,
                    "gold_text": gold_row.get("span_text") if gold_row else None,
                    "error_category": _categorize_predicted_span(predicted_uri, gold_row, match_type, candidates),
                }
            )

        for gold_index in sorted(unmatched_gold):
            gold_row = posting_gold_rows[gold_index]
            span_text = str(gold_row.get("span_text") or "")
            candidates = rank_span_candidates(
                span_text,
                ranker,
                lexicon_index=lexicon_index,
                top_k=top_k,
            )
            posting_traces.append(
                {
                    "record_type": "missed_gold_span",
                    "posting_id": posting_id,
                    "source_id": record.get("source_id"),
                    "role_type": record.get("source"),
                    "sentence": _best_sentence(record, gold_row),
                    "span_start": gold_row.get("span_start"),
                    "span_end": gold_row.get("span_end"),
                    "span": span_text,
                    "span_confidence": 0.0,
                    "ner_label": gold_row.get("span_label"),
                    "top5_candidates": candidates,
                    "predicted_uri": None,
                    "gold_uri": gold_row.get("gold_concept_uri"),
                    "gold_label": gold_row.get("span_label"),
                    "gold_span_start": gold_row.get("span_start"),
                    "gold_span_end": gold_row.get("span_end"),
                    "gold_text": span_text,
                    "error_category": "missed_extraction",
                }
            )

        traces[posting_id] = posting_traces

    return traces


def write_audit_traces(traces: dict[str, list[dict[str, object]]], output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    total_records = 0
    error_counts: Counter[str] = Counter()

    for posting_id, records in traces.items():
        output_path = output_dir / f"{posting_id}.jsonl"
        with output_path.open("w", encoding="utf-8") as fh:
            for record in records:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_records += 1
                error_counts[str(record.get("error_category") or "unknown")] += 1

    manifest = {
        "output_dir": str(output_dir),
        "postings": len(traces),
        "trace_files": len(traces),
        "trace_records": total_records,
        "error_counts": dict(sorted(error_counts.items())),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def generate_audit_traces(
    ner_report: Path = DEFAULT_NER_REPORT,
    review_csv: Path = DEFAULT_REVIEW_CSV,
    esco_csv: Path = DEFAULT_ESCO_CSV,
    embeddings: Path = DEFAULT_EMBEDDINGS,
    index: Path = DEFAULT_INDEX,
    labels: Path = DEFAULT_LABELS,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    top_k: int = 5,
) -> dict[str, object]:
    gold_report = _load_json(ner_report)
    gold_rows = _load_review_rows(review_csv)
    ranker = load_span_ranker(embeddings, index, labels)
    lexicon_index = load_lexicon_index(esco_csv)
    traces = build_audit_trace_records(gold_report["document_records"], gold_rows, ranker, lexicon_index, top_k=top_k)
    manifest = write_audit_traces(traces, output_dir)
    manifest["gold_documents"] = gold_report["documents"]
    manifest["gold_source"] = gold_report["source"]
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Write end-to-end audit traces for the gold corpus.")
    parser.add_argument("--ner-report", type=Path, default=DEFAULT_NER_REPORT)
    parser.add_argument("--review-csv", type=Path, default=DEFAULT_REVIEW_CSV)
    parser.add_argument("--esco-csv", type=Path, default=DEFAULT_ESCO_CSV)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    manifest = generate_audit_traces(
        ner_report=args.ner_report,
        review_csv=args.review_csv,
        esco_csv=args.esco_csv,
        embeddings=args.embeddings,
        index=args.index,
        labels=args.labels,
        output_dir=args.output_dir,
        top_k=args.top_k,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
