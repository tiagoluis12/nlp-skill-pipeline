import json
from pathlib import Path

from src.evaluation.build_ablation_table import build_ablation_rows, write_ablation_table


def _write_report(results_dir: Path, name: str, payload: dict[str, object]) -> None:
    (results_dir / f"ablation_{name.lower()}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_build_ablation_rows_preserves_delta_semantics(tmp_path: Path) -> None:
    for name in ("A1", "A2"):
        _write_report(
            tmp_path,
            name,
            {
                "ablation": name,
                "description": f"description {name}",
                "delta_vs_full_pipeline": {
                    "span_exact_f1": -0.1,
                    "esco_acc_at_1": -0.2,
                    "esco_acc_at_3": -0.3,
                    "mrr": -0.4,
                },
            },
        )
    _write_report(
        tmp_path,
        "A3",
        {
            "ablation": "A3",
            "description": "domain transfer",
            "domain_transfer_gap": {"span_exact": {"f1": 0.25}},
            "interpretation": "transfer gap",
        },
    )
    _write_report(
        tmp_path,
        "A4",
        {
            "ablation": "A4",
            "description": "embedding swap",
            "delta_minilm_minus_mpnet": {"acc_at_1": 0.01, "acc_at_3": 0.02, "mrr": 0.03},
            "interpretation": "MiniLM improved linking",
        },
    )

    rows = build_ablation_rows(tmp_path)

    assert [row["ablation"] for row in rows] == ["A1", "A2", "A3", "A4"]
    assert rows[0]["span_f1_delta"] == "-0.100000"
    assert rows[2]["delta_reference"] == "SKILLSPAN test minus gold standard"
    assert rows[2]["esco_acc1_delta"] == "N/A"
    assert rows[3]["span_f1_delta"] == "0.000000"
    assert rows[3]["esco_acc1_delta"] == "0.010000"


def test_write_ablation_table_writes_header_and_rows(tmp_path: Path) -> None:
    output = tmp_path / "ablation_table.csv"
    write_ablation_table([{"ablation": "A1", "config_description": "x", "delta_reference": "y", "span_f1_delta": "0", "esco_acc1_delta": "0", "esco_acc3_delta": "0", "mrr_delta": "0", "interpretation": "z"}], output)

    content = output.read_text(encoding="utf-8")
    assert content.startswith("ablation,config_description,delta_reference,span_f1_delta")
    assert "A1,x,y,0,0,0,0,z" in content
