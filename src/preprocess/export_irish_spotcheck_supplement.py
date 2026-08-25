"""Export one real, previously unreviewed QA posting for spot-check coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_tasks(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [payload] if isinstance(payload, dict) else payload


def build_supplement(main_path: Path, existing_spotcheck_path: Path) -> list[dict]:
    main_tasks = load_tasks(main_path)
    existing_tasks = load_tasks(existing_spotcheck_path)
    existing_ids = {str((task.get("data") or {}).get("source_id")) for task in existing_tasks}
    candidates = [
        task
        for task in main_tasks
        if (task.get("data") or {}).get("role_type") == "qa_engineer"
        and str((task.get("data") or {}).get("source_id")) not in existing_ids
    ]
    if not candidates:
        raise ValueError("No unreviewed QA task is available for the supplementary spot-check")
    return [candidates[0]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a real QA task for Irish spot-check coverage.")
    parser.add_argument("--main", type=Path, default=Path("data/annotated/irish_gold_remaining_labelstudio.json"))
    parser.add_argument("--existing-spotcheck", type=Path, default=Path("data/annotated/irish_gold_spotcheck_labelstudio.json"))
    parser.add_argument("--output", type=Path, default=Path("data/annotated/irish_gold_spotcheck_qa_labelstudio.json"))
    args = parser.parse_args()
    tasks = build_supplement(args.main, args.existing_spotcheck)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(tasks, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"tasks": len(tasks), "source_id": tasks[0]["data"]["source_id"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
