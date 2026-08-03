from src.evaluation.run_ablation_a4 import _deltas
import pytest


def test_a4_delta_is_minilm_minus_mpnet() -> None:
    mpnet = {"acc_at_1": 0.50, "acc_at_3": 0.80, "mrr": 0.65, "macro_f1": 0.47}
    minilm = {"acc_at_1": 0.45, "acc_at_3": 0.75, "mrr": 0.60, "macro_f1": 0.42}

    deltas = _deltas(minilm, mpnet)

    assert deltas == pytest.approx(
        {"acc_at_1": -0.05, "acc_at_3": -0.05, "mrr": -0.05, "macro_f1": -0.05}
    )
