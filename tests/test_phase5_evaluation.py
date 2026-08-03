from src.evaluation.evaluate_phase5 import _coverage_from_records, _row_from_report


def test_coverage_requires_exact_span_and_uri_match() -> None:
    records = [
        {
            "id": 10,
            "predicted_spans": [
                {"start": 1, "end": 3, "concept_uri": "uri-correct"},
                {"start": 4, "end": 5, "concept_uri": "uri-wrong"},
            ],
        },
        {"id": 11, "predicted_spans": [{"start": 1, "end": 4, "concept_uri": "uri-correct"}]},
    ]
    review = {("10", 1, 3): "uri-correct", ("11", 1, 3): "uri-correct"}

    assert _coverage_from_records(records, review) == (1, 2)


def test_skillspan_row_has_no_linking_metrics() -> None:
    report = {
        "documents": 2,
        "token_metrics": {"precision": 0.1, "recall": 0.2, "f1": 0.13},
        "span_exact": {"precision": 0.3, "recall": 0.4, "f1": 0.34},
        "span_relaxed": {"precision": 0.5, "recall": 0.6, "f1": 0.55},
    }
    row = _row_from_report("Baseline1_Lexicon", "skillspan_test", report, None, None)

    assert row["esco_acc_at_1"] == ""
    assert row["end_to_end_coverage"] == ""
    assert "N/A" in row["notes"]
