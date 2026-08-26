from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path
from typing import Sequence

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from src.esco.ranker import load_lexicon_index, load_span_ranker, rank_span_candidates_batch
from src.extraction.ner_model import load_checkpoint_bundle, predict_text_spans


DEFAULT_CLEAN_DIR = Path("data/clean")
DEFAULT_GOLD_JSONL = Path("data/annotated/gold_annotator1.normalized.jsonl")
DEFAULT_OUTPUT_DIR = Path("results/integration_traces")
DEFAULT_REPORT = Path("results/integration_report.json")
DEFAULT_ESCO_CSV = Path("data/benchmarks/esco/skills_en.csv")
DEFAULT_EMBEDDINGS = Path("models/esco_embeddings.npy")
DEFAULT_INDEX = Path("models/esco_faiss.index")
DEFAULT_LABELS = Path("models/esco_embedding_labels.json")
DEFAULT_PER_ROLE = 4
ROLE_ORDER = ("cloud_support", "cybersecurity", "data_analyst", "qa_engineer", "software_dev")
INTEGRATION_RANKER_MODEL_NAME = "all-mpnet-base-v2"


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_gold_ids(gold_jsonl: Path = DEFAULT_GOLD_JSONL) -> set[str]:
    return {str(row.get("id")) for row in _load_jsonl(gold_jsonl)}


def select_unseen_postings(
    clean_dir: Path = DEFAULT_CLEAN_DIR,
    gold_jsonl: Path = DEFAULT_GOLD_JSONL,
    per_role: int = DEFAULT_PER_ROLE,
    input_jsonl: Path | None = None,
) -> list[dict[str, object]]:
    gold_ids = load_gold_ids(gold_jsonl)
    selected: list[dict[str, object]] = []
    consolidated_records = _load_jsonl(input_jsonl) if input_jsonl is not None else None

    for role_type in ROLE_ORDER:
        if consolidated_records is None:
            path = clean_dir / f"{role_type}.jsonl"
            source_records = _load_jsonl(path)
        else:
            source_records = [row for row in consolidated_records if row.get("role_type") == role_type]
        records = [row for row in source_records if str(row.get("id")) not in gold_ids]
        records.sort(key=lambda row: str(row.get("id")))
        if len(records) < per_role:
            raise ValueError(f"Not enough unseen records for {role_type}: need {per_role}, found {len(records)}")
        selected.extend(records[:per_role])

    return selected


def _candidate_summary(candidates: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "rank": idx + 1,
            "label": candidate.get("label"),
            "concept_uri": candidate.get("concept_uri"),
            "score": float(candidate.get("score") or 0.0),
            "method": candidate.get("method"),
        }
        for idx, candidate in enumerate(candidates)
    ]


def _sentence_for_span(sentences: Sequence[str], span_text: str, raw_text: str) -> str:
    span_lower = span_text.lower().strip()
    for sentence in sentences:
        if span_lower and span_lower in sentence.lower():
            return sentence.strip()
    return raw_text.strip()


def run_integration_record(
    record: dict[str, object],
    bundle,
    ranker,
    lexicon_index: dict[str, str],
    top_k: int = 5,
) -> dict[str, object]:
    raw_text = str(record.get("raw_text") or "")
    sentences = [str(sentence) for sentence in record.get("sentences") or []]
    spans = predict_text_spans(bundle, raw_text)
    ranked_candidates = rank_span_candidates_batch(
        [span.text for span in spans],
        ranker,
        lexicon_index=lexicon_index,
        top_k=top_k,
    )

    trace_spans: list[dict[str, object]] = []
    uri_populated = 0
    for span in spans:
        candidates = ranked_candidates.get(span.text, [])
        predicted_uri = candidates[0]["concept_uri"] if candidates else None
        if predicted_uri:
            uri_populated += 1
        trace_spans.append(
            {
                "posting_id": record.get("id"),
                "source_id": record.get("id"),
                "role_type": record.get("role_type"),
                "raw_text": raw_text,
                "sentence": _sentence_for_span(sentences, span.text, raw_text),
                "span": span.text,
                "span_confidence": span.confidence,
                "ner_label": span.label,
                "top5_candidates": _candidate_summary(candidates),
                "predicted_uri": predicted_uri,
                "error_category": "uri_missing" if not predicted_uri else "integration_ok",
            }
        )

    failure_reasons: list[str] = []
    if not spans:
        failure_reasons.append("no_spans_detected")
    if spans and uri_populated == 0:
        failure_reasons.append("no_esco_uri_populated")

    return {
        "posting_id": record.get("id"),
        "role_type": record.get("role_type"),
        "span_count": len(spans),
        "uri_populated": uri_populated,
        "failure_reasons": failure_reasons,
        "traces": trace_spans,
    }


def write_integration_outputs(
    records: Sequence[dict[str, object]],
    bundle,
    ranker,
    lexicon_index: dict[str, str],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    top_k: int = 5,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    total_spans = 0
    total_uris = 0
    failures: list[dict[str, object]] = []
    error_counts: Counter[str] = Counter()

    for record in records:
        report = run_integration_record(record, bundle, ranker, lexicon_index, top_k=top_k)
        posting_path = output_dir / f"{report['posting_id']}.jsonl"
        with posting_path.open("w", encoding="utf-8") as fh:
            for trace in report["traces"]:
                fh.write(json.dumps(trace, ensure_ascii=False) + "\n")

        total_spans += int(report["span_count"])
        total_uris += int(report["uri_populated"])
        if report["failure_reasons"]:
            failures.append(
                {
                    "posting_id": report["posting_id"],
                    "role_type": report["role_type"],
                    "failure_reasons": report["failure_reasons"],
                }
            )
            for reason in report["failure_reasons"]:
                error_counts[reason] += 1

    manifest = {
        "postings": len(records),
        "trace_files": len(records),
        "trace_records": total_spans,
        "predicted_uris": total_uris,
        "failures": failures,
        "failure_count": len(failures),
        "error_counts": dict(sorted(error_counts.items())),
        "output_dir": str(output_dir),
    }

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def run_integration_gate(
    clean_dir: Path = DEFAULT_CLEAN_DIR,
    gold_jsonl: Path = DEFAULT_GOLD_JSONL,
    esco_csv: Path = DEFAULT_ESCO_CSV,
    embeddings: Path = DEFAULT_EMBEDDINGS,
    index: Path = DEFAULT_INDEX,
    labels: Path = DEFAULT_LABELS,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    report_output: Path = DEFAULT_REPORT,
    per_role: int = DEFAULT_PER_ROLE,
    top_k: int = 5,
    input_jsonl: Path | None = None,
) -> dict[str, object]:
    bundle = load_checkpoint_bundle()
    ranker = load_span_ranker(
        embeddings,
        index,
        labels,
        model_name=INTEGRATION_RANKER_MODEL_NAME,
        allow_fallback=False,
    )
    lexicon_index = load_lexicon_index(esco_csv)

    records = select_unseen_postings(
        clean_dir=clean_dir,
        gold_jsonl=gold_jsonl,
        per_role=per_role,
        input_jsonl=input_jsonl,
    )
    manifest = write_integration_outputs(records, bundle, ranker, lexicon_index, output_dir=output_dir, top_k=top_k)
    manifest["selected_roles"] = list(ROLE_ORDER)
    manifest["per_role"] = per_role
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Phase 4 integration gate on 20 unseen postings.")
    parser.add_argument("--clean-dir", type=Path, default=DEFAULT_CLEAN_DIR)
    parser.add_argument("--gold-jsonl", type=Path, default=DEFAULT_GOLD_JSONL)
    parser.add_argument("--esco-csv", type=Path, default=DEFAULT_ESCO_CSV)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--per-role", type=int, default=DEFAULT_PER_ROLE)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--input-jsonl", type=Path, default=None, help="Consolidated corpus to use instead of role-separated files.")
    args = parser.parse_args()

    manifest = run_integration_gate(
        clean_dir=args.clean_dir,
        gold_jsonl=args.gold_jsonl,
        esco_csv=args.esco_csv,
        embeddings=args.embeddings,
        index=args.index,
        labels=args.labels,
        output_dir=args.output_dir,
        report_output=args.report_output,
        per_role=args.per_role,
        top_k=args.top_k,
        input_jsonl=args.input_jsonl,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
