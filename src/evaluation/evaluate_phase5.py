"""Run the frozen Phase 5.1 evaluation and write the final comparison table.

Extraction reports are read from the frozen Phase 3/4 artefacts. Baseline 2 is
re-evaluated on SKILLSPAN test with the threshold selected on development data
so that the test split is not used for threshold selection.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from datasets import load_from_disk
from sklearn.metrics import f1_score

from src.esco.ranker import (
    load_span_ranker,
    rank_candidate_texts,
    rank_document_candidates,
    rank_span_candidates,
    select_predictions,
)
from src.evaluation.evaluate_baseline1 import (
    Span,
    _bio_to_spans,
    _labels_from_skillspan,
)
from src.evaluation.evaluate_baseline2 import run_threshold_search
from src.extraction.lexicon_baseline import build_lexicon_index, load_esco_skills, match_span_detail


SYSTEMS = ("Baseline1_Lexicon", "Baseline2_Embedding", "Proposed_NER_Ranking")
DATASETS = ("skillspan_test", "gold")
DEFAULT_SKILLSPAN_DIR = Path("data/benchmarks/skillspan")
DEFAULT_GOLD_JSONL = Path("data/annotated/gold_annotator1.normalized.jsonl")
DEFAULT_REVIEW_CSV = Path("results/gold_esco_review_priority.csv")
DEFAULT_ESCO_CSV = Path("data/benchmarks/esco/skills_en.csv")
DEFAULT_OUTPUT = Path("results/final_results_table.csv")
DEFAULT_LINKING_OUTPUT = Path("results/phase5_esco_linking.json")
DEFAULT_BASELINE1_PREDICTIONS = Path("results/baseline1_predictions.jsonl")
DEFAULT_AUDIT_DIR = Path("results/audit_traces")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _float_or_blank(value: Any) -> float | str:
    return "" if value is None else float(value)


def _metric(report: dict[str, Any], section: str, key: str) -> float | str:
    section_data = report.get(section) or {}
    value = section_data.get(key)
    return _float_or_blank(value)


def _classification_values(report: dict[str, Any], label: str) -> dict[str, float | str]:
    per_label = ((report.get("classification_report") or {}).get("per_label") or {}).get(label) or {}
    return {
        "precision": _float_or_blank(per_label.get("precision")),
        "recall": _float_or_blank(per_label.get("recall")),
        "f1": _float_or_blank(per_label.get("f1")),
    }


def _span_dicts(labels: Sequence[str]) -> list[dict[str, Any]]:
    return [
        {"start": span.start, "end": span.end, "label": span.label}
        for span in _bio_to_spans(labels)
    ]


def _gold_rows(gold_jsonl: Path) -> list[dict[str, Any]]:
    return _read_jsonl(gold_jsonl)


def _baseline2_gold_records(
    gold_rows: Sequence[dict[str, Any]],
    ranker: Any,
    threshold: float,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in gold_rows:
        candidates = rank_document_candidates(row["tokens"], ranker)
        selected = select_predictions(candidates, threshold)
        records.append(
            {
                "id": row["id"],
                "gold_spans": _span_dicts(row["labels"]),
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
    return records


def _coverage_from_records(
    records: Iterable[dict[str, Any]],
    review_by_span: dict[tuple[str, int, int], str],
) -> tuple[int, int]:
    """Return postings with one exact span+URI match and total postings."""
    correct_postings: set[str] = set()
    all_postings: set[str] = set()
    for record in records:
        posting_id = str(record.get("id") or record.get("posting_id"))
        if not posting_id:
            continue
        all_postings.add(posting_id)
        predicted_spans = record.get("predicted_spans") or []
        for predicted in predicted_spans:
            key = (posting_id, int(predicted["start"]), int(predicted["end"]))
            gold_uri = review_by_span.get(key)
            if gold_uri and predicted.get("concept_uri") == gold_uri:
                correct_postings.add(posting_id)
                break
    return len(correct_postings), len(all_postings)


def _coverage_from_audit(audit_dir: Path) -> tuple[int, int]:
    correct_postings: set[str] = set()
    postings: set[str] = set()
    for path in sorted(audit_dir.glob("*.jsonl")):
        if path.stem.isdigit():
            postings.add(path.stem)
        for record in _read_jsonl(path):
            posting_id = str(record.get("posting_id", ""))
            if posting_id:
                postings.add(posting_id)
            if record.get("record_type") != "predicted_span":
                continue
            if not posting_id:
                continue
            gold_uri = record.get("gold_uri")
            if not gold_uri or record.get("predicted_uri") != gold_uri:
                continue
            if (
                str(record.get("span_start")) == str(record.get("gold_span_start"))
                and str(record.get("span_end")) == str(record.get("gold_span_end"))
            ):
                correct_postings.add(posting_id)
    return len(correct_postings), len(postings)


def _review_index(review_csv: Path) -> tuple[list[dict[str, str]], dict[tuple[str, int, int], str]]:
    with review_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    index: dict[tuple[str, int, int], str] = {}
    for row in rows:
        uri = (row.get("gold_concept_uri") or "").strip()
        if not uri:
            continue
        index[(str(row["posting_id"]), int(row["span_start"]), int(row["span_end"]))] = uri
    return rows, index


def _link_predictions(
    rows: Sequence[dict[str, str]],
    system: str,
    ranker: Any,
    lexicon_index: dict[str, str],
) -> tuple[dict[str, float | int], list[dict[str, str]]]:
    gold_rows = [row for row in rows if (row.get("gold_concept_uri") or "").strip()]
    gold_uris: list[str] = []
    predicted_uris: list[str] = []
    details: list[dict[str, str]] = []
    acc1 = 0
    acc3 = 0
    mrr_total = 0.0

    for row in gold_rows:
        gold_uri = row["gold_concept_uri"].strip()
        text = row["span_text"]
        if system == "Baseline1_Lexicon":
            result = match_span_detail(text, lexicon_index)
            predictions = [result.concept_uri] if result.concept_uri else []
        elif system == "Baseline2_Embedding":
            _, indices = rank_candidate_texts([text], ranker, top_k=5)
            predictions = [ranker.artifacts.concept_uris[index] for index in indices[0] if index >= 0]
        elif system == "Proposed_NER_Ranking":
            predictions = [
                str(candidate["concept_uri"])
                for candidate in rank_span_candidates(text, ranker, lexicon_index=lexicon_index, top_k=5)
            ]
        else:
            raise ValueError(f"Unknown system: {system}")

        hit1 = gold_uri in predictions[:1]
        hit3 = gold_uri in predictions[:3]
        rank = predictions.index(gold_uri) + 1 if gold_uri in predictions else None
        acc1 += int(hit1)
        acc3 += int(hit3)
        mrr_total += 1.0 / rank if rank else 0.0
        gold_uris.append(gold_uri)
        predicted_uris.append(predictions[0] if predictions else "__NO_PREDICTION__")
        details.append(
            {
                "posting_id": row["posting_id"],
                "span_text": text,
                "gold_uri": gold_uri,
                "predicted_uri": predictions[0] if predictions else "",
                "gold_label": row.get("gold_label", ""),
                "predicted_label": row.get("candidate_1_label", "") if predictions else "",
            }
        )

    total = len(gold_rows)
    macro_f1 = f1_score(
        gold_uris,
        predicted_uris,
        labels=sorted(set(gold_uris)),
        average="macro",
        zero_division=0,
    ) if total else None
    metrics: dict[str, float | int] = {
        "documents": total,
        "acc_at_1": acc1 / total if total else 0.0,
        "acc_at_3": acc3 / total if total else 0.0,
        "mrr": mrr_total / total if total else 0.0,
        "macro_f1": float(macro_f1) if macro_f1 is not None else 0.0,
    }
    return metrics, details


def _load_extraction_reports(results_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    reports = {
        ("Baseline1_Lexicon", "skillspan_test"): _read_json(results_dir / "baseline1_skillspan.json"),
        ("Baseline1_Lexicon", "gold"): _read_json(results_dir / "baseline1_gold.json"),
        ("Baseline2_Embedding", "gold"): _read_json(results_dir / "baseline2_gold.json"),
        ("Proposed_NER_Ranking", "skillspan_test"): _read_json(results_dir / "ner_skillspan.json"),
        ("Proposed_NER_Ranking", "gold"): _read_json(results_dir / "ner_gold.json"),
    }
    return reports


def _row_from_report(
    system: str,
    dataset: str,
    report: dict[str, Any],
    linking: dict[str, float | int] | None,
    coverage: tuple[int, int] | None,
) -> dict[str, Any]:
    tech = _classification_values(report, "SKILL_TECH")
    trans = _classification_values(report, "SKILL_TRANS")
    row: dict[str, Any] = {
        "system": system,
        "dataset": dataset,
        "documents": report.get("documents", ""),
        "token_precision": _metric(report, "token_metrics", "precision"),
        "token_recall": _metric(report, "token_metrics", "recall"),
        "token_f1": _metric(report, "token_metrics", "f1"),
        "span_exact_precision": _metric(report, "span_exact", "precision"),
        "span_exact_recall": _metric(report, "span_exact", "recall"),
        "span_exact_f1": _metric(report, "span_exact", "f1"),
        "span_relaxed_precision": _metric(report, "span_relaxed", "precision"),
        "span_relaxed_recall": _metric(report, "span_relaxed", "recall"),
        "span_relaxed_f1": _metric(report, "span_relaxed", "f1"),
        "skill_tech_precision": tech["precision"],
        "skill_tech_recall": tech["recall"],
        "skill_tech_f1": tech["f1"],
        "skill_trans_precision": trans["precision"],
        "skill_trans_recall": trans["recall"],
        "skill_trans_f1": trans["f1"],
        "esco_acc_at_1": "",
        "esco_acc_at_3": "",
        "esco_mrr": "",
        "esco_macro_f1": "",
        "esco_evaluation_units": "",
        "end_to_end_coverage": "",
        "end_to_end_correct_postings": "",
        "end_to_end_postings": "",
        "notes": "",
    }
    if linking is not None:
        row.update(
            {
                "esco_acc_at_1": linking["acc_at_1"],
                "esco_acc_at_3": linking["acc_at_3"],
                "esco_mrr": linking["mrr"],
                "esco_macro_f1": linking["macro_f1"],
                "esco_evaluation_units": linking["documents"],
            }
        )
    if coverage is not None:
        correct, total = coverage
        row.update(
            {
                "end_to_end_coverage": correct / total if total else 0.0,
                "end_to_end_correct_postings": correct,
                "end_to_end_postings": total,
            }
        )
    if system != "Proposed_NER_Ranking":
        row["notes"] = "Class-agnostic extractor; per-class NER metrics are not applicable."
    if dataset == "skillspan_test":
        row["notes"] += " ESCO metrics and end-to-end coverage are N/A because reviewed ESCO gold URIs are unavailable."
    else:
        row["notes"] += " ESCO metrics use 601 reviewed gold spans; coverage requires exact span and URI agreement."
    return row


def run_phase5_evaluation(
    skillspan_dir: Path = DEFAULT_SKILLSPAN_DIR,
    gold_jsonl: Path = DEFAULT_GOLD_JSONL,
    review_csv: Path = DEFAULT_REVIEW_CSV,
    esco_csv: Path = DEFAULT_ESCO_CSV,
    results_dir: Path = Path("results"),
    embeddings: Path = Path("models/esco_embeddings.npy"),
    index: Path = Path("models/esco_faiss.index"),
    labels: Path = Path("models/esco_embedding_labels.json"),
    audit_dir: Path = DEFAULT_AUDIT_DIR,
    output_csv: Path = DEFAULT_OUTPUT,
    linking_output: Path = DEFAULT_LINKING_OUTPUT,
) -> list[dict[str, Any]]:
    threshold_payload = _read_json(results_dir / "b2_threshold_search.json")
    dev_threshold = float(threshold_payload["best_threshold"])
    ranker = load_span_ranker(embeddings, index, labels)
    if ranker.artifacts.model_name.endswith("-fallback-hashing"):
        raise RuntimeError(
            "Phase 5 evaluation requires a local sentence-transformer model; "
            "the hashing fallback is not valid for final reported metrics."
        )

    test_payload = run_threshold_search(skillspan_dir, "test", ranker, [dev_threshold])
    baseline2_test_output = results_dir / "baseline2_skillspan.json"
    baseline2_test_output.write_text(json.dumps(test_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    test_report = dict(test_payload["thresholds"][0])
    test_report.update({"source": "skillspan", "split": "test", "documents": test_payload["documents"]})

    reports = _load_extraction_reports(results_dir)
    reports[("Baseline2_Embedding", "skillspan_test")] = test_report

    review_rows, review_by_span = _review_index(review_csv)
    skills = load_esco_skills(esco_csv)
    lexicon_index = build_lexicon_index(skills)
    linking_reports: dict[str, dict[str, float | int]] = {}
    linking_details: dict[str, list[dict[str, str]]] = {}
    for system in SYSTEMS:
        metrics, details = _link_predictions(review_rows, system, ranker, lexicon_index)
        linking_reports[system] = metrics
        linking_details[system] = details

    gold_rows = _gold_rows(gold_jsonl)
    baseline1_prediction_rows = [
        row for row in _read_jsonl(DEFAULT_BASELINE1_PREDICTIONS) if row.get("role_type") is not None
    ]
    baseline1_coverage_records = [
        {
            "id": row["id"],
            "predicted_spans": row.get("predicted_spans", []),
        }
        for row in baseline1_prediction_rows
    ]
    baseline2_gold_records = _baseline2_gold_records(gold_rows, ranker, dev_threshold)
    proposed_coverage = _coverage_from_audit(audit_dir)
    coverage = {
        ("Baseline1_Lexicon", "gold"): _coverage_from_records(baseline1_coverage_records, review_by_span),
        ("Baseline2_Embedding", "gold"): _coverage_from_records(baseline2_gold_records, review_by_span),
        ("Proposed_NER_Ranking", "gold"): proposed_coverage,
    }

    rows: list[dict[str, Any]] = []
    for system in SYSTEMS:
        for dataset in DATASETS:
            report = reports[(system, dataset)]
            linking = linking_reports[system] if dataset == "gold" else None
            row_coverage = coverage.get((system, dataset))
            rows.append(_row_from_report(system, dataset, report, linking, row_coverage))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    linking_output.parent.mkdir(parents=True, exist_ok=True)
    linking_output.write_text(
        json.dumps(
            {
                "review_csv": str(review_csv),
                "reviewed_rows": len(review_rows),
                "systems": linking_reports,
                "threshold_source": "results/b2_threshold_search.json",
                "baseline2_test_threshold": dev_threshold,
                "retrieval": {
                    "model": ranker.artifacts.model_name,
                    "index": type(ranker.artifacts.index).__name__,
                    "embedding_shape": list(ranker.artifacts.embeddings.shape),
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Phase 5.1 full evaluation.")
    parser.add_argument("--skillspan-dir", type=Path, default=DEFAULT_SKILLSPAN_DIR)
    parser.add_argument("--gold-jsonl", type=Path, default=DEFAULT_GOLD_JSONL)
    parser.add_argument("--review-csv", type=Path, default=DEFAULT_REVIEW_CSV)
    parser.add_argument("--esco-csv", type=Path, default=DEFAULT_ESCO_CSV)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--linking-output", type=Path, default=DEFAULT_LINKING_OUTPUT)
    args = parser.parse_args()
    rows = run_phase5_evaluation(
        skillspan_dir=args.skillspan_dir,
        gold_jsonl=args.gold_jsonl,
        review_csv=args.review_csv,
        esco_csv=args.esco_csv,
        results_dir=args.results_dir,
        output_csv=args.output,
        linking_output=args.linking_output,
    )
    print(json.dumps({"rows": len(rows), "output": str(args.output)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
