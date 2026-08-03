"""Build the Phase 5.9 top-10 ESCO misclassification figure."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from textwrap import wrap
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont

from src.esco.ranker import load_lexicon_index, load_span_ranker, rank_span_candidates
from src.extraction.lexicon_baseline import load_esco_skills


DEFAULT_REVIEW_CSV = Path("results/gold_esco_review_priority.csv")
DEFAULT_ESCO_CSV = Path("data/benchmarks/esco/skills_en.csv")
DEFAULT_OUTPUT = Path("docs/figures/confusion_matrix.png")
DEFAULT_METADATA = Path("results/confusion_matrix.json")


def _read_review_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _uri_labels(esco_csv: Path) -> dict[str, str]:
    skills = load_esco_skills(esco_csv)
    return {
        str(row["conceptUri"]): str(row["preferredLabel"])
        for _, row in skills.iterrows()
        if str(row["conceptUri"]).strip()
    }


def summarize_misclassifications(
    ranking_results: Iterable[dict[str, Any]],
    uri_labels: dict[str, str],
) -> dict[str, Any]:
    pair_counts: Counter[tuple[str, str]] = Counter()
    reviewed = 0
    correct_top1 = 0
    wrong_top1 = 0
    no_prediction = 0
    coverage_gap_excluded = 0

    for result in ranking_results:
        gold_uri = str(result.get("gold_uri") or "")
        predicted_uri = str(result.get("predicted_uri") or "")
        top5_uris = [str(uri) for uri in result.get("top5_uris", []) if uri]
        if not gold_uri:
            continue
        reviewed += 1
        if not predicted_uri:
            no_prediction += 1
        elif predicted_uri == gold_uri:
            correct_top1 += 1
            continue
        else:
            wrong_top1 += 1

        # A pair is a ranking error only when the gold concept was retrieved.
        if predicted_uri and gold_uri in top5_uris:
            pair_counts[(predicted_uri, gold_uri)] += 1
        elif predicted_uri:
            coverage_gap_excluded += 1

    top_pairs = [
        {
            "predicted_uri": predicted_uri,
            "gold_uri": gold_uri,
            "predicted_label": uri_labels.get(predicted_uri, predicted_uri),
            "gold_label": uri_labels.get(gold_uri, gold_uri),
            "count": count,
        }
        for (predicted_uri, gold_uri), count in pair_counts.most_common(10)
    ]
    return {
        "reviewed_spans": reviewed,
        "correct_top1": correct_top1,
        "wrong_top1": wrong_top1,
        "no_prediction": no_prediction,
        "coverage_gap_excluded": coverage_gap_excluded,
        "misclassification_pairs": top_pairs,
        "all_pair_count": sum(pair_counts.values()),
    }


def collect_ranking_results(
    review_rows: list[dict[str, str]],
    ranker: Any,
    lexicon_index: dict[str, str],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for row in review_rows:
        candidates = rank_span_candidates(
            row["span_text"],
            ranker,
            lexicon_index=lexicon_index,
            top_k=5,
        )
        results.append(
            {
                "gold_uri": row.get("gold_concept_uri", ""),
                "predicted_uri": candidates[0]["concept_uri"] if candidates else "",
                "top5_uris": [candidate["concept_uri"] for candidate in candidates],
            }
        )
    return results


def write_figure(summary: dict[str, Any], output_path: Path) -> None:
    pairs = summary["misclassification_pairs"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    font_path = "/System/Library/Fonts/Supplemental/Arial.ttf"
    try:
        title_font = ImageFont.truetype(font_path, 34)
        body_font = ImageFont.truetype(font_path, 22)
        small_font = ImageFont.truetype(font_path, 18)
    except OSError:
        title_font = body_font = small_font = ImageFont.load_default()

    cell_width = 240
    left = 210
    right = 80
    top = 210
    bottom = 190
    width = max(1400, left + right + cell_width * max(1, len(pairs)))
    height = 900
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((left, 35), "Top-10 ESCO Acc@1 Misclassification Pairs", fill="black", font=title_font)
    draw.text((left, 85), "Proposed NER + MPNet Ranking", fill="#444444", font=body_font)

    if pairs:
        max_count = max(int(pair["count"]) for pair in pairs)
        cell_height = 190
        for index, pair in enumerate(pairs):
            x0 = left + index * cell_width
            x1 = x0 + cell_width - 8
            intensity = int(255 - 150 * int(pair["count"]) / max_count) if max_count else 255
            draw.rectangle((x0, top, x1, top + cell_height), fill=(255, intensity, intensity), outline="#8b0000", width=2)
            count_text = str(pair["count"])
            count_box = draw.textbbox((0, 0), count_text, font=title_font)
            draw.text((x0 + (cell_width - 8 - (count_box[2] - count_box[0])) / 2, top + 65), count_text, fill="black", font=title_font)
            label = f"{pair['predicted_label']} -> {pair['gold_label']}"
            lines = wrap(label, width=25)
            draw.multiline_text((x0 + 8, top + cell_height + 18), "\n".join(lines), fill="black", font=small_font, spacing=4)
        draw.text((40, top + 72), "Wrong\ntop-1\ncount", fill="black", font=small_font, spacing=4)
    else:
        draw.text((left, top + 80), "No eligible misclassification pairs", fill="black", font=body_font)

    footer = (
        f"Reviewed spans: {summary['reviewed_spans']} | Wrong top-1: {summary['wrong_top1']} | "
        f"Coverage-gap cases excluded: {summary['coverage_gap_excluded']}"
    )
    draw.text((left, height - 70), footer, fill="#444444", font=small_font)
    image.save(output_path, dpi=(300, 300))


def build_confusion_matrix(
    review_csv: Path = DEFAULT_REVIEW_CSV,
    esco_csv: Path = DEFAULT_ESCO_CSV,
    output_path: Path = DEFAULT_OUTPUT,
    metadata_path: Path = DEFAULT_METADATA,
) -> dict[str, Any]:
    review_rows = _read_review_rows(review_csv)
    ranker = load_span_ranker()
    if ranker.artifacts.model_name.endswith("-fallback-hashing"):
        raise RuntimeError("The misclassification figure requires the corrected local MPNet ranker.")
    lexicon_index = load_lexicon_index(esco_csv)
    uri_labels = _uri_labels(esco_csv)
    ranking_results = collect_ranking_results(review_rows, ranker, lexicon_index)
    summary = summarize_misclassifications(ranking_results, uri_labels)
    summary["retrieval"] = {
        "model": ranker.artifacts.model_name,
        "index": type(ranker.artifacts.index).__name__,
        "embedding_shape": list(ranker.artifacts.embeddings.shape),
    }
    write_figure(summary, output_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Phase 5.9 top-10 ESCO misclassification figure.")
    parser.add_argument("--review-csv", type=Path, default=DEFAULT_REVIEW_CSV)
    parser.add_argument("--esco-csv", type=Path, default=DEFAULT_ESCO_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    args = parser.parse_args()
    summary = build_confusion_matrix(args.review_csv, args.esco_csv, args.output, args.metadata)
    print(json.dumps({"output": str(args.output), "pairs": len(summary["misclassification_pairs"])}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
