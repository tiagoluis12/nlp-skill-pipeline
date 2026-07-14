from __future__ import annotations

import argparse
from pathlib import Path

from src.preprocess.clean import clean_text, load_sentence_nlp


def clean_file(input_path: Path, output_path: Path) -> dict[str, int]:
    import json

    nlp = load_sentence_nlp()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    total = 0
    kept = 0

    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            total += 1
            record = json.loads(line)
            raw_text = record.get("description") or record.get("raw_text") or ""
            sentences = clean_text(raw_text, nlp)
            cleaned_text = " ".join(sentences)
            if not cleaned_text or cleaned_text in seen:
                continue
            seen.add(cleaned_text)
            dst.write(
                json.dumps(
                    {
                        "id": record.get("id"),
                        "role_type": record.get("role_type"),
                        "country": record.get("country"),
                        "sentences": sentences,
                        "raw_text": raw_text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            kept += 1

    summary = {"input_rows": total, "output_rows": kept, "dedup_removed": total - kept}
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean all raw role files into sentence-segmented JSONL.")
    parser.add_argument("--input-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/clean"))
    args = parser.parse_args()

    for input_path in sorted(args.input_dir.glob("*.jsonl")):
        if input_path.name == ".gitkeep":
            continue
        output_path = args.output_dir / input_path.name
        clean_file(input_path, output_path)


if __name__ == "__main__":
    main()
