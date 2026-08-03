"""Build the Phase 5.7 error taxonomy from end-to-end audit traces."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_AUDIT_DIR = Path("results/audit_traces")
DEFAULT_OUTPUT = Path("results/error_taxonomy.md")

CATEGORY_SPECS: dict[str, dict[str, str]] = {
    "E1": {
        "name": "Boundary Error",
        "source": "boundary_error",
        "why": "The extractor identifies the right local context but includes or excludes tokens, punctuation, or neighbouring words.",
        "fix": "Use boundary-aware span decoding, train with more multi-token examples, and add post-processing for punctuation and noun-phrase boundaries.",
    },
    "E2": {
        "name": "ESCO Coverage Gap",
        "source": "esco_coverage_gap",
        "why": "The gold concept is not present among the retrieved candidates, indicating terminology or granularity that the current ESCO label index does not represent well.",
        "fix": "Add reviewed aliases and descriptions, improve candidate retrieval, and document genuine ESCO taxonomy gaps instead of forcing an incorrect URI.",
    },
    "E3": {
        "name": "Soft-Skill Ambiguity",
        "source": "soft_skill_ambiguity",
        "why": "Soft skills are semantically broad and overlapping, so a correctly extracted SKILL_TRANS span can map to a neighbouring ESCO concept.",
        "fix": "Use label-aware contextual reranking, soft-skill-specific hard negatives, and adjudicated contextual examples.",
    },
    "E4": {
        "name": "Other ESCO Linking Error",
        "source": "esco_linking_error",
        "why": "The correct concept is available to the ranker, but another candidate receives the highest score.",
        "fix": "Use hard-negative ranking examples, cross-encoder reranking, and concept-description features to distinguish closely related ESCO skills.",
    },
    "E5": {
        "name": "False-Positive Extraction",
        "source": "false_positive",
        "why": "The NER model labels generic job-posting language or contextual words as skills even though the gold annotation contains no matching span.",
        "fix": "Add hard negative examples, calibrate the confidence threshold, and apply role/title and generic-language filtering.",
    },
    "E6": {
        "name": "Missed Skill Extraction",
        "source": "missed_extraction",
        "why": "A gold skill span is not produced by the extractor, reflecting domain shift, terminology coverage, or low-confidence NER decisions.",
        "fix": "Add in-domain training data, use domain-adaptive pretraining or augmentation, and inspect low-confidence spans during active learning.",
    },
}

SOURCE_TO_CODE = {spec["source"]: code for code, spec in CATEGORY_SPECS.items()}


def iter_trace_records(audit_dir: Path) -> Iterable[dict[str, Any]]:
    for path in sorted(audit_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                yield json.loads(line)


def _example(record: dict[str, Any]) -> dict[str, str]:
    return {
        "posting_id": str(record.get("posting_id") or ""),
        "role_type": str(record.get("role_type") or ""),
        "span": str(record.get("span") or record.get("gold_text") or ""),
        "gold_text": str(record.get("gold_text") or ""),
        "sentence": str(record.get("sentence") or ""),
    }


def analyse_traces(records: list[dict[str, Any]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    role_counts: dict[str, Counter[str]] = defaultdict(Counter)
    examples: dict[str, list[dict[str, str]]] = defaultdict(list)
    gold_labels = Counter(str(record.get("gold_label") or "") for record in records if record.get("gold_uri"))

    for record in records:
        source_category = str(record.get("error_category") or "unknown")
        if source_category == "correct":
            continue
        code = SOURCE_TO_CODE.get(source_category, "E7")
        counts[code] += 1
        role = str(record.get("role_type") or "unknown")
        role_counts[role][code] += 1
        if len(examples[code]) < 3:
            examples[code].append(_example(record))

    error_records = sum(counts.values())
    return {
        "total_records": len(records),
        "correct_records": sum(1 for record in records if record.get("error_category") == "correct"),
        "error_records": error_records,
        "counts": dict(counts),
        "frequencies": {
            code: (count / error_records if error_records else 0.0) for code, count in counts.items()
        },
        "role_counts": {role: dict(counter) for role, counter in sorted(role_counts.items())},
        "examples": dict(examples),
        "gold_labels": dict(gold_labels),
    }


def _escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ").strip()


def _render_examples(examples: list[dict[str, str]]) -> list[str]:
    lines: list[str] = []
    for example in examples:
        sentence = _escape(example["sentence"])
        if len(sentence) > 240:
            sentence = sentence[:237] + "..."
        lines.append(
            f"- Posting `{_escape(example['posting_id'])}` ({_escape(example['role_type'])}): "
            f"`{_escape(example['span'])}`; gold text `{_escape(example['gold_text'])}`. "
            f"Context: {sentence}"
        )
    return lines


def render_markdown(analysis: dict[str, Any]) -> str:
    error_records = int(analysis["error_records"])
    lines = [
        "# Error Taxonomy",
        "",
        "This taxonomy was generated from `results/audit_traces/` after the corrected MPNet evaluation.",
        "",
        f"- Total trace records: **{analysis['total_records']}**",
        f"- Correct records: **{analysis['correct_records']}**",
        f"- Error records: **{error_records}**",
        "- Frequencies below use all error records as the denominator.",
        "",
        "## Summary",
        "",
        "| Code | Category | Count | Frequency of errors |",
        "|---|---|---:|---:|",
    ]

    for code, spec in CATEGORY_SPECS.items():
        count = int(analysis["counts"].get(code, 0))
        frequency = float(analysis["frequencies"].get(code, 0.0))
        lines.append(f"| {code} | {spec['name']} | {count} | {frequency:.2%} |")

    unknown_count = int(analysis["counts"].get("E7", 0))
    if unknown_count:
        lines.append(f"| E7 | Other / unclassified | {unknown_count} | {unknown_count / error_records:.2%} |")

    lines.extend(["", "## Interpretation and Corrective Actions", ""])
    for code, spec in CATEGORY_SPECS.items():
        count = int(analysis["counts"].get(code, 0))
        lines.extend([f"### {code} — {spec['name']}", "", f"- Count: **{count}**"])
        if code == "E3" and count == 0:
            lines.append(
                "- Data limitation: no reviewed gold spans were labelled `SKILL_TRANS`, so this category cannot be estimated from the current gold corpus."
            )
        lines.extend([f"- Why it occurs: {spec['why']}", f"- Proposed fix: {spec['fix']}", ""])
        if analysis["examples"].get(code):
            lines.append("Representative examples:")
            lines.extend(_render_examples(analysis["examples"][code]))
            lines.append("")

    lines.extend(["## Role-Type Breakdown", "", "| Role type | " + " | ".join(CATEGORY_SPECS) + " |", "|---|" + "---:|" * len(CATEGORY_SPECS)])
    for role, role_counts in analysis["role_counts"].items():
        values = " | ".join(str(role_counts.get(code, 0)) for code in CATEGORY_SPECS)
        lines.append(f"| {_escape(role)} | {values} |")

    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "The taxonomy is based on the 125-posting gold corpus and its reviewed ESCO links. E3 is included as a required category, but its frequency is zero because the reviewed corpus contains no `SKILL_TRANS` gold spans. The results therefore support conclusions about boundary, extraction, and technical ESCO-linking errors more strongly than conclusions about soft-skill ambiguity.",
            "",
        ]
    )
    return "\n".join(lines)


def build_error_taxonomy(audit_dir: Path = DEFAULT_AUDIT_DIR, output_path: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    analysis = analyse_traces(list(iter_trace_records(audit_dir)))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_markdown(analysis), encoding="utf-8")
    return analysis


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Phase 5.7 error taxonomy.")
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    analysis = build_error_taxonomy(args.audit_dir, args.output)
    print(json.dumps({"output": str(args.output), "error_records": analysis["error_records"], "counts": analysis["counts"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
