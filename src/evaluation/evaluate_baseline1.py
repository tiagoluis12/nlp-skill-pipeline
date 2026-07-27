from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from datasets import load_from_disk
from seqeval.metrics import f1_score, precision_score, recall_score

from src.extraction.lexicon_baseline import (
    build_lexicon_index,
    build_token_index,
    candidate_labels_for_span,
    load_esco_skills,
    match_span_detail,
    normalize_text,
)


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    label: str
    score: float = 0.0
    concept_uri: str | None = None


def load_gold_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def _bio_to_spans(labels: Sequence[str]) -> list[Span]:
    spans: list[Span] = []
    start = None
    current_label = None
    for idx, label in enumerate(list(labels) + ["O"]):
        if label.startswith("B-"):
            if start is not None and current_label is not None:
                spans.append(Span(start=start, end=idx, label=current_label))
            start = idx
            current_label = label[2:]
        elif label.startswith("I-") and current_label == label[2:]:
            continue
        else:
            if start is not None and current_label is not None:
                spans.append(Span(start=start, end=idx, label=current_label))
            start = None
            current_label = None
    return spans


def _spans_to_bio(length: int, spans: Sequence[Span]) -> list[str]:
    labels = ["O"] * length
    for span in spans:
        if span.start >= span.end:
            continue
        labels[span.start] = f"B-{span.label}"
        for idx in range(span.start + 1, span.end):
            labels[idx] = f"I-{span.label}"
    return labels


def _span_overlap(a: Span, b: Span) -> bool:
    return min(a.end, b.end) > max(a.start, b.start)


def _score_spans_exact(gold: Sequence[Span], pred: Sequence[Span]) -> dict[str, float]:
    gold_set = {(span.start, span.end, span.label) for span in gold}
    pred_set = {(span.start, span.end, span.label) for span in pred}
    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def _score_spans_relaxed(gold: Sequence[Span], pred: Sequence[Span]) -> dict[str, float]:
    gold_by_label: dict[str, list[int]] = {}
    for idx, span in enumerate(gold):
        gold_by_label.setdefault(span.label, []).append(idx)

    matched_gold: set[int] = set()
    tp = 0
    fp = 0
    for pred_span in pred:
        candidates = gold_by_label.get(pred_span.label, [])
        match_idx = None
        for gold_idx in candidates:
            if gold_idx in matched_gold:
                continue
            if _span_overlap(pred_span, gold[gold_idx]):
                match_idx = gold_idx
                break
        if match_idx is None:
            fp += 1
        else:
            matched_gold.add(match_idx)
            tp += 1
    fn = len(gold) - len(matched_gold)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def _labels_from_skillspan(row: dict) -> list[str]:
    skill = row.get("tags_skill") or []
    knowledge = row.get("tags_knowledge") or []
    if len(skill) != len(knowledge):
        raise ValueError("SKILLSPAN token/label length mismatch")

    labels: list[str] = []
    for skill_tag, knowledge_tag in zip(skill, knowledge, strict=True):
        if skill_tag != "O":
            labels.append(f"{skill_tag}-SKILL_TECH")
        elif knowledge_tag != "O":
            labels.append(f"{knowledge_tag}-SKILL_TRANS")
        else:
            labels.append("O")
    return labels


def _predict_document_spans(tokens: Sequence[str], index: dict[str, str], token_index: dict[str, set[str]], max_ngram: int = 5, score_cutoff: int = 85) -> list[Span]:
    spans: list[Span] = []
    i = 0
    while i < len(tokens):
        best_exact: Span | None = None
        best_exact_len = 0
        for length in range(min(max_ngram, len(tokens) - i), 0, -1):
            phrase = " ".join(tokens[i : i + length])
            normalized = normalize_text(phrase)
            if normalized in index:
                best_exact = Span(start=i, end=i + length, label="SKILL_TECH", score=100.0, concept_uri=index[normalized])
                best_exact_len = length
                break

        if best_exact is not None:
            spans.append(best_exact)
            i += best_exact_len
            continue

        best_fuzzy: Span | None = None
        best_fuzzy_len = 0
        for length in range(min(max_ngram, len(tokens) - i), 0, -1):
            phrase = " ".join(tokens[i : i + length])
            candidates = candidate_labels_for_span(phrase, index, token_index=token_index)
            if not candidates:
                continue
            result = match_span_detail(phrase, index, score_cutoff=score_cutoff, candidate_labels=candidates)
            if result.concept_uri is None:
                continue
            candidate = Span(start=i, end=i + length, label="SKILL_TECH", score=result.score, concept_uri=result.concept_uri)
            if best_fuzzy is None or candidate.score > best_fuzzy.score or (candidate.score == best_fuzzy.score and length > best_fuzzy_len):
                best_fuzzy = candidate
                best_fuzzy_len = length

        if best_fuzzy is not None:
            spans.append(best_fuzzy)
            i += best_fuzzy_len
        else:
            i += 1

    return spans


def _evaluate_documents(records: Iterable[dict], source: str, index: dict[str, str], token_index: dict[str, set[str]]) -> dict[str, object]:
    gold_labels_all: list[list[str]] = []
    pred_labels_all: list[list[str]] = []
    gold_spans_all: list[Span] = []
    pred_spans_all: list[Span] = []
    documents: list[dict[str, object]] = []

    for row in records:
        tokens = row["tokens"]
        if source == "skillspan":
            gold_labels = _labels_from_skillspan(row)
        else:
            gold_labels = row["labels"]

        gold_spans = _bio_to_spans(gold_labels)
        pred_spans = _predict_document_spans(tokens, index, token_index)
        pred_labels = _spans_to_bio(len(tokens), pred_spans)

        gold_labels_all.append(gold_labels)
        pred_labels_all.append(pred_labels)
        gold_spans_all.extend(gold_spans)
        pred_spans_all.extend(pred_spans)
        documents.append(
            {
                "id": row.get("id"),
                "source_id": row.get("source_id"),
                "role_type": row.get("role_type"),
                "gold_spans": [span.__dict__ for span in gold_spans],
                "predicted_spans": [span.__dict__ for span in pred_spans],
            }
        )

    token_metrics = {
        "precision": precision_score(gold_labels_all, pred_labels_all),
        "recall": recall_score(gold_labels_all, pred_labels_all),
        "f1": f1_score(gold_labels_all, pred_labels_all),
    }
    exact = _score_spans_exact(gold_spans_all, pred_spans_all)
    relaxed = _score_spans_relaxed(gold_spans_all, pred_spans_all)

    return {
        "source": source,
        "documents": len(documents),
        "token_metrics": token_metrics,
        "span_exact": exact,
        "span_relaxed": relaxed,
        "esco_metrics": None,
        "document_records": documents,
    }


def run_skillspan_evaluation(skillspan_disk: Path, split: str, esco_csv: Path) -> dict[str, object]:
    dataset = load_from_disk(str(skillspan_disk))
    if split not in dataset:
        raise KeyError(f"Split {split!r} not found in {skillspan_disk}")
    records = [
        {
            "id": row["idx"],
            "tokens": row["tokens"],
            "tags_skill": row["tags_skill"],
            "tags_knowledge": row["tags_knowledge"],
        }
        for row in dataset[split]
    ]

    skills = load_esco_skills(esco_csv)
    index = build_lexicon_index(skills)
    token_index = build_token_index(index)
    return _evaluate_documents(records, "skillspan", index, token_index)


def run_gold_evaluation(gold_jsonl: Path, esco_csv: Path) -> dict[str, object]:
    records = load_gold_jsonl(gold_jsonl)
    skills = load_esco_skills(esco_csv)
    index = build_lexicon_index(skills)
    token_index = build_token_index(index)
    return _evaluate_documents(records, "gold", index, token_index)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the ESCO lexicon baseline on SKILLSPAN and the gold set.")
    parser.add_argument("--skillspan-disk", type=Path, default=Path("data/benchmarks/skillspan"))
    parser.add_argument("--skillspan-split", default="test")
    parser.add_argument("--gold-jsonl", type=Path, default=Path("data/annotated/gold_annotator1.normalized.jsonl"))
    parser.add_argument("--esco-csv", type=Path, default=Path("data/benchmarks/esco/skills_en.csv"))
    parser.add_argument("--skillspan-output", type=Path, default=Path("results/baseline1_skillspan.json"))
    parser.add_argument("--gold-output", type=Path, default=Path("results/baseline1_gold.json"))
    parser.add_argument("--predictions-output", type=Path, default=None)
    args = parser.parse_args()

    skillspan_report = run_skillspan_evaluation(args.skillspan_disk, args.skillspan_split, args.esco_csv)
    gold_report = run_gold_evaluation(args.gold_jsonl, args.esco_csv)

    def serializable(report: dict[str, object]) -> dict[str, object]:
        return {key: value for key, value in report.items() if key != "document_records"}

    args.skillspan_output.parent.mkdir(parents=True, exist_ok=True)
    args.skillspan_output.write_text(json.dumps(serializable(skillspan_report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.gold_output.parent.mkdir(parents=True, exist_ok=True)
    args.gold_output.write_text(json.dumps(serializable(gold_report), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.predictions_output is not None:
        args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
        with args.predictions_output.open("w", encoding="utf-8") as fh:
            for report in (skillspan_report, gold_report):
                for document in report["document_records"]:
                    fh.write(json.dumps(document, ensure_ascii=False) + "\n")

    print(json.dumps({"skillspan": serializable(skillspan_report), "gold": serializable(gold_report)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
