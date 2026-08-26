from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from datasets import load_from_disk
from seqeval.metrics import f1_score, precision_score, recall_score

from src.esco.ranker import load_span_ranker, rank_document_candidates, select_predictions
from src.evaluation.evaluate_baseline1 import _score_spans_exact, _score_spans_relaxed


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


def _spans_to_bio(length: int, spans: Sequence[dict[str, int | str]]) -> list[str]:
    labels = ["O"] * length
    for span in spans:
        start = int(span["start"])
        end = int(span["end"])
        label = str(span["label"])
        if start >= end:
            continue
        labels[start] = f"B-{label}"
        for idx in range(start + 1, end):
            labels[idx] = f"I-{label}"
    return labels


def _skillspan_gold_spans(row: dict) -> list[dict[str, int | str]]:
    spans: list[dict[str, int | str]] = []
    for field in ("tags_skill", "tags_knowledge"):
        labels = row.get(field) or []
        if not labels:
            continue
        for span in _bio_to_spans([f"{tag}-SKILL" if tag != "O" else "O" for tag in labels]):
            spans.append(span)
    return spans


def _row_to_gold_spans(row: dict) -> list[dict[str, int | str]]:
    labels = row.get("labels") or []
    if any(label.startswith("B-") or label.startswith("I-") for label in labels):
        return _bio_to_spans([label.replace("SKILL_TECH", "SKILL").replace("SKILL_TRANS", "SKILL") for label in labels])
    return []


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _evaluate_threshold(records: list[dict], ranker, threshold: float, token_pred_label: str = "SKILL") -> dict[str, object]:
    gold_all: list[dict[str, int | str]] = []
    pred_all: list[dict[str, int | str]] = []
    gold_token_all: list[list[str]] = []
    pred_token_all: list[list[str]] = []

    candidate_total = 0
    for row in records:
        gold_spans = row["gold_spans"]
        candidates = row["candidates"]
        selected = select_predictions(candidates, threshold)
        pred_spans = [{"start": c.start, "end": c.end, "label": "SKILL"} for c in selected]
        candidate_total += len(candidates)
        gold_all.extend(gold_spans)
        pred_all.extend(pred_spans)
        gold_token_all.append(row["gold_labels"])
        pred_token_all.append(
            _spans_to_bio(
                len(row["tokens"]),
                [{"start": span["start"], "end": span["end"], "label": token_pred_label} for span in pred_spans],
            )
        )
    from src.evaluation.evaluate_baseline1 import Span

    gold_span_objs = [Span(start=int(span["start"]), end=int(span["end"]), label=str(span["label"])) for span in gold_all]
    pred_span_objs = [Span(start=int(span["start"]), end=int(span["end"]), label=str(span["label"])) for span in pred_all]

    token_metrics = {
        "precision": precision_score(gold_token_all, pred_token_all),
        "recall": recall_score(gold_token_all, pred_token_all),
        "f1": f1_score(gold_token_all, pred_token_all),
    }
    exact = _score_spans_exact(gold_span_objs, pred_span_objs)
    relaxed = _score_spans_relaxed(gold_span_objs, pred_span_objs)
    return {
        "threshold": threshold,
        "documents": len(records),
        "candidate_total": candidate_total,
        "token_metrics": token_metrics,
        "span_exact": exact,
        "span_relaxed": relaxed,
    }


def run_threshold_search(
    skillspan_disk: Path,
    split: str,
    ranker,
    thresholds: Sequence[float],
) -> dict[str, object]:
    thresholds = list(dict.fromkeys(thresholds))
    dataset = load_from_disk(str(skillspan_disk))
    if split not in dataset:
        raise KeyError(f"Split {split!r} not found in {skillspan_disk}")
    records = [
        {
            "id": row["idx"],
            "tokens": row["tokens"],
            "tags_skill": row["tags_skill"],
            "tags_knowledge": row["tags_knowledge"],
            "role_type": row.get("source"),
        }
        for row in dataset[split]
    ]

    prepared_records = []
    for row in records:
        gold_spans = _skillspan_gold_spans(row) if "tags_skill" in row else _row_to_gold_spans(row)
        prepared_records.append(
            {
                "id": row["id"],
                "tokens": row["tokens"],
                "gold_spans": gold_spans,
                "gold_labels": row["labels"] if "labels" in row else _spans_to_bio(len(row["tokens"]), gold_spans),
                "candidates": rank_document_candidates(row["tokens"], ranker),
            }
        )

    threshold_results = [_evaluate_threshold(prepared_records, ranker, threshold) for threshold in thresholds]
    best = max(threshold_results, key=lambda result: result["span_exact"]["f1"])
    payload = {
        "split": split,
        "documents": len(prepared_records),
        "thresholds": threshold_results,
        "best_threshold": best["threshold"],
        "best_span_exact_f1": best["span_exact"]["f1"],
    }
    return payload


def run_gold_evaluation(
    gold_jsonl: Path,
    ranker,
    threshold: float,
    predictions_output: Path | None = None,
) -> dict[str, object]:
    records = [
        {
            "id": row["id"],
            "source_id": row.get("source_id"),
            "role_type": row.get("role_type"),
            "tokens": row["tokens"],
            "labels": row["labels"],
        }
        for row in _load_jsonl(gold_jsonl)
    ]

    prepared_records = []
    prediction_records = []
    for row in records:
        candidates = rank_document_candidates(row["tokens"], ranker)
        selected = select_predictions(candidates, threshold)
        prepared_records.append(
            {
                "id": row["id"],
                "tokens": row["tokens"],
                "gold_spans": _row_to_gold_spans(row),
                "gold_labels": row["labels"],
                "candidates": candidates,
            }
        )
        prediction_records.append(
            {
                "id": row["id"],
                "source_id": row.get("source_id"),
                "role_type": row.get("role_type"),
                "predicted_spans": [
                    {
                        "start": candidate.start,
                        "end": candidate.end,
                        "label": "SKILL_TECH",
                        "concept_uri": candidate.predicted_uri,
                    }
                    for candidate in selected
                ],
            }
        )

    result = _evaluate_threshold(prepared_records, ranker, threshold, token_pred_label="SKILL_TECH")
    result["source"] = "gold"
    if predictions_output is not None:
        predictions_output.parent.mkdir(parents=True, exist_ok=True)
        with predictions_output.open("w", encoding="utf-8") as handle:
            for record in prediction_records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune the embedding baseline threshold on SKILLSPAN dev.")
    parser.add_argument("--skillspan-disk", type=Path, default=Path("data/benchmarks/skillspan"))
    parser.add_argument("--split", default="dev")
    parser.add_argument("--output", type=Path, default=Path("results/b2_threshold_search.json"))
    parser.add_argument("--gold-jsonl", type=Path, default=None)
    parser.add_argument("--gold-output", type=Path, default=Path("results/baseline2_gold.json"))
    parser.add_argument("--gold-predictions-output", type=Path, default=None)
    parser.add_argument("--embeddings", type=Path, default=Path("models/esco_embeddings.npy"))
    parser.add_argument("--index", type=Path, default=Path("models/esco_faiss.index"))
    parser.add_argument("--labels", type=Path, default=Path("models/esco_embedding_labels.json"))
    parser.add_argument("--model-name", default="all-mpnet-base-v2")
    parser.add_argument("--threshold", type=float, action="append", default=None)
    parser.add_argument("--skip-threshold-search", action="store_true")
    args = parser.parse_args()

    ranker = load_span_ranker(args.embeddings, args.index, args.labels, model_name=args.model_name)
    if args.skip_threshold_search:
        if args.threshold is None or len(args.threshold) != 1:
            raise ValueError("--skip-threshold-search requires exactly one --threshold value")
        payload = {"best_threshold": args.threshold[0], "split": args.split}
    else:
        thresholds = args.threshold if args.threshold is not None else [0.60, 0.65, 0.70, 0.75]
        payload = run_threshold_search(args.skillspan_disk, args.split, ranker, thresholds)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2, sort_keys=True))

    if args.gold_jsonl is not None:
        gold_payload = run_gold_evaluation(
            args.gold_jsonl,
            ranker,
            float(payload["best_threshold"]),
            predictions_output=args.gold_predictions_output,
        )
        args.gold_output.parent.mkdir(parents=True, exist_ok=True)
        args.gold_output.write_text(json.dumps(gold_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
