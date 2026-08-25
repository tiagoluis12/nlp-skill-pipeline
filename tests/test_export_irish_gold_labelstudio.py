import json
from pathlib import Path

from src.preprocess.export_irish_gold_labelstudio import build_exports


def test_build_exports_excludes_pilot_and_creates_spot_check(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text(
        "\n".join(
            json.dumps({"id": f"{role}-{index}", "role_type": role, "sentences": ["Python required"]})
            for role in ["cloud_support", "software_dev"]
            for index in range(2)
        )
        + "\n",
        encoding="utf-8",
    )
    pilot = tmp_path / "pilot.json"
    pilot.write_text(json.dumps([{"data": {"source_id": "cloud_support-0"}}]), encoding="utf-8")

    tasks, spot_check = build_exports(source, pilot, spot_fraction=0.5, seed=42)

    assert len(tasks) == 3
    assert all(task["data"]["source_id"] != "cloud_support-0" for task in tasks)
    assert len(spot_check) == 2
    assert {task["data"]["source_id"] for task in spot_check}.issubset(
        {task["data"]["source_id"] for task in tasks}
    )
