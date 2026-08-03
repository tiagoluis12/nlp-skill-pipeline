from src.evaluation.run_ablation_a3 import _transfer_gaps


def test_transfer_gap_is_in_distribution_minus_out_of_distribution() -> None:
    in_report = {
        "token_metrics": {"precision": 0.8, "recall": 0.7, "f1": 0.75},
        "span_exact": {"precision": 0.8, "recall": 0.6, "f1": 0.685714},
        "span_relaxed": {"precision": 0.9, "recall": 0.8, "f1": 0.847058},
        "classification_report": {
            "per_label": {
                "SKILL_TECH": {"precision": 0.8, "recall": 0.7, "f1": 0.75},
                "SKILL_TRANS": {"precision": 0.6, "recall": 0.5, "f1": 0.55},
            }
        },
    }
    out_report = {
        "token_metrics": {"precision": 0.3, "recall": 0.2, "f1": 0.24},
        "span_exact": {"precision": 0.4, "recall": 0.3, "f1": 0.342857},
        "span_relaxed": {"precision": 0.6, "recall": 0.5, "f1": 0.545454},
        "classification_report": {
            "per_label": {
                "SKILL_TECH": {"precision": 0.3, "recall": 0.2, "f1": 0.24},
                "SKILL_TRANS": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            }
        },
    }

    gaps = _transfer_gaps(in_report, out_report)

    assert gaps["token_metrics"]["f1"] == 0.51
    assert gaps["per_class"]["SKILL_TRANS"]["f1"] == 0.55
