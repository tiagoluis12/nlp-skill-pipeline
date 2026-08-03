from src.evaluation.build_confusion_matrix import summarize_misclassifications


def test_summary_excludes_coverage_gaps_from_pair_counts() -> None:
    summary = summarize_misclassifications(
        [
            {"gold_uri": "gold-1", "predicted_uri": "pred-1", "top5_uris": ["pred-1", "gold-1"]},
            {"gold_uri": "gold-2", "predicted_uri": "pred-2", "top5_uris": ["pred-2"]},
            {"gold_uri": "gold-3", "predicted_uri": "gold-3", "top5_uris": ["gold-3"]},
        ],
        {"pred-1": "Predicted one", "gold-1": "Gold one"},
    )

    assert summary["reviewed_spans"] == 3
    assert summary["correct_top1"] == 1
    assert summary["wrong_top1"] == 2
    assert summary["coverage_gap_excluded"] == 1
    assert summary["all_pair_count"] == 1
    assert summary["misclassification_pairs"][0]["predicted_label"] == "Predicted one"


def test_summary_counts_no_prediction_without_creating_pair() -> None:
    summary = summarize_misclassifications(
        [{"gold_uri": "gold-1", "predicted_uri": "", "top5_uris": []}],
        {"gold-1": "Gold one"},
    )

    assert summary["no_prediction"] == 1
    assert summary["all_pair_count"] == 0
