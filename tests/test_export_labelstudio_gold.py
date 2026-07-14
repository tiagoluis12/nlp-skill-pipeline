from __future__ import annotations

import json

from src.preprocess.export_labelstudio_gold import load_excluded_ids, make_spot_check, select_stratified_records


def _write_jsonl(path, rows) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def test_export_labelstudio_gold_excludes_pilot_and_keeps_roles(tmp_path) -> None:
    clean_dir = tmp_path / "clean"
    clean_dir.mkdir()

    _write_jsonl(
        clean_dir / "software_dev.jsonl",
        [
            {"id": "s1", "role_type": "software_dev", "sentences": ["A."]},
            {"id": "s2", "role_type": "software_dev", "sentences": ["B."]},
            {"id": "s3", "role_type": "software_dev", "sentences": ["C."]},
        ],
    )
    _write_jsonl(
        clean_dir / "data_analyst.jsonl",
        [
            {"id": "d1", "role_type": "data_analyst", "sentences": ["D."]},
            {"id": "d2", "role_type": "data_analyst", "sentences": ["E."]},
            {"id": "d3", "role_type": "data_analyst", "sentences": ["F."]},
        ],
    )

    pilot_path = tmp_path / "pilot.json"
    pilot_path.write_text(
        json.dumps(
            [
                {"data": {"source_id": "s1"}},
                {"data": {"source_id": "d1"}},
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    excluded_ids = load_excluded_ids(pilot_path)
    gold = select_stratified_records(clean_dir, per_role=2, excluded_ids=excluded_ids, seed=42)

    assert excluded_ids == {"s1", "d1"}
    assert len(gold) == 4
    assert {row["data"]["role_type"] for row in gold} == {"software_dev", "data_analyst"}
    assert {row["data"]["source_id"] for row in gold} == {"s2", "s3", "d2", "d3"}

    spot_check = make_spot_check(gold, fraction=0.25, seed=42)
    assert len(spot_check) == 1
    assert spot_check[0] in gold
