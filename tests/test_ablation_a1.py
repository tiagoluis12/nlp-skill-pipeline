from src.evaluation.run_ablation_a1 import _deltas


def test_a1_extraction_delta_is_zero_when_ner_is_unchanged() -> None:
    report = {
        "extraction": {
            "span_exact": {"f1": 0.36},
            "span_relaxed": {"f1": 0.616},
        },
        "esco_linking": {"acc_at_1": 0.22, "acc_at_3": 0.22, "mrr": 0.22},
        "end_to_end": {"coverage": 0.296},
    }
    full = {
        "span_exact_f1": "0.36",
        "span_relaxed_f1": "0.616",
        "esco_acc_at_1": "0.51",
        "esco_acc_at_3": "0.81",
        "esco_mrr": "0.66",
        "end_to_end_coverage": "0.072",
    }

    deltas = _deltas(report, full)

    assert deltas["span_exact_f1"] == 0.0
    assert deltas["span_relaxed_f1"] == 0.0
    assert deltas["esco_acc_at_1"] < 0.0

