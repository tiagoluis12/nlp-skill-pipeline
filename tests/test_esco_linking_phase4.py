from __future__ import annotations

import csv
import json
from pathlib import Path

import src.evaluation.evaluate_esco_linking as linking
from src.evaluation.update_baseline_comparison import refresh_baseline_comparison


def test_evaluate_rows_supports_proposed(monkeypatch) -> None:
    rows = [
        {
            "gold_concept_uri": "uri-1",
            "span_text": "Python testing",
        }
    ]

    monkeypatch.setattr(
        linking,
        "rank_span_candidates",
        lambda span_text, ranker, lexicon_index=None, top_k=5: [
            {"concept_uri": "uri-1"},
            {"concept_uri": "uri-2"},
        ],
    )

    report = linking._evaluate_rows(rows, "proposed", ranker=object(), lexicon_index={})

    assert report["documents"] == 1
    assert report["acc_at_1"] == 1.0
    assert report["acc_at_3"] == 1.0
    assert report["mrr"] == 1.0


def test_refresh_baseline_comparison_updates_rows(tmp_path: Path) -> None:
    comparison_csv = tmp_path / "baseline_comparison.csv"
    comparison_csv.write_text(
        "system,dataset,documents,token_precision,token_recall,token_f1,span_exact_precision,span_exact_recall,span_exact_f1,span_relaxed_precision,span_relaxed_recall,span_relaxed_f1,esco_acc_at_1,esco_acc_at_3,esco_mrr,notes\n"
        "Baseline1_Lexicon,old,0,,,,,,,,,,,,old note\n"
        "Baseline2_Embedding,old,0,,,,,,,,,,,,old note\n"
        "Proposed_Pipeline,skillspan,0,,,,,,,,,,,,old note\n",
        encoding="utf-8",
    )

    extraction = {
        "source": "gold",
        "documents": 2,
        "token_metrics": {"precision": 0.1, "recall": 0.2, "f1": 0.15},
        "span_exact": {"precision": 0.3, "recall": 0.4, "f1": 0.35},
        "span_relaxed": {"precision": 0.5, "recall": 0.6, "f1": 0.55},
    }
    linking_metrics = {"acc_at_1": 0.7, "acc_at_3": 0.8, "mrr": 0.75}

    for name in ["baseline1", "baseline2", "proposed"]:
        (tmp_path / f"{name}_extraction.json").write_text(json.dumps(extraction), encoding="utf-8")
        (tmp_path / f"{name}_linking.json").write_text(json.dumps(linking_metrics), encoding="utf-8")

    output_csv = tmp_path / "out.csv"
    refresh_baseline_comparison(
        comparison_csv=comparison_csv,
        output_csv=output_csv,
        baseline1_extraction=tmp_path / "baseline1_extraction.json",
        baseline2_extraction=tmp_path / "baseline2_extraction.json",
        proposed_extraction=tmp_path / "proposed_extraction.json",
        baseline1_linking=tmp_path / "baseline1_linking.json",
        baseline2_linking=tmp_path / "baseline2_linking.json",
        proposed_linking=tmp_path / "proposed_linking.json",
    )

    rows = list(csv.DictReader(output_csv.read_text(encoding="utf-8").splitlines()))
    assert rows[0]["dataset"] == "gold"
    assert rows[0]["esco_acc_at_1"] == "0.7"
    assert rows[2]["notes"] == "NER evaluated on gold corpus; ESCO linking on reviewed gold spans"
