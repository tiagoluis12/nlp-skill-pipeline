"""Create transparent, role-specific queues for human Irish-corpus review."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.preprocess.screen_irish_corpus import ROLE_TYPES

TITLE_PATTERNS = {
    "cloud_support": re.compile(r"\b(?:it support|technical support|service desk|help ?desk|cloud|devops|infrastructure|systems? engineer|network support)\b", re.I),
    "cybersecurity": re.compile(r"\b(?:cyber|security|infosec|soc|forensics|penetration)\b", re.I),
    "data_analyst": re.compile(r"\b(?:data analyst|business intelligence|bi developer|reporting analyst|analytics)\b", re.I),
    "qa_engineer": re.compile(r"\b(?:qa|quality assurance|test automation|test engineer|software test|sdet|testing engineer)\b", re.I),
    "software_dev": re.compile(r"\b(?:software (?:engineer|developer)|application(?:s)? (?:engineer|developer)|(?:front|back)end|full[ -]?stack|\.net developer|java developer|python developer)\b", re.I),
}
NON_TECH_TITLE_PATTERNS = {
    "cloud_support": re.compile(r"\b(?:civil|electrical|environmental|structural|sales|fund|distribution|financial)\b", re.I),
    "cybersecurity": re.compile(r"\b(?:product owner|sales|audit)\b", re.I),
    "data_analyst": re.compile(r"\b(?:funds? operations|pricing|finance|investment|risk)\b", re.I),
    "qa_engineer": re.compile(r"\b(?:welding|construction|animal|biotech|pharma|food|mechanical|electrical|inspection|qc)\b", re.I),
    "software_dev": re.compile(r"\b(?:construction|cnc|machine operator|welding)\b", re.I),
}
ENTRY_DESCRIPTION_RE = re.compile(r"\b(?:0[- ]?2|1[- ]?2|up to two) years? (?:of )?(?:professional )?experience\b", re.I)
EXPERIENCE_RE = re.compile(r"\b(?:[3-9]|[1-9]\d)\+? years?(?: of)? experience\b", re.I)
AI_TRAINER_RE = re.compile(r"\bai trainer\b", re.I)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def rank_record(record: dict[str, Any], role_type: str) -> tuple[int, list[str]]:
    title = record["title"]
    text = record["raw_text"][:3000]
    score = 0
    reasons: list[str] = []
    if TITLE_PATTERNS[role_type].search(title):
        score += 6
        reasons.append("title_matches_target_role")
    if record.get("role_confidence") == "high":
        score += 3
        reasons.append("high_role_confidence")
    elif record.get("role_confidence") == "medium":
        score += 1
    if record.get("entry_signal"):
        score += 5
        reasons.append("entry_title_signal")
    if ENTRY_DESCRIPTION_RE.search(text):
        score += 3
        reasons.append("entry_experience_signal")
    if NON_TECH_TITLE_PATTERNS[role_type].search(title):
        score -= 7
        reasons.append("non_tech_title_signal")
    if EXPERIENCE_RE.search(text):
        score -= 4
        reasons.append("experience_requirement_3_plus_years")
    if AI_TRAINER_RE.search(title):
        score -= 6
        reasons.append("ai_trainer_listing")
    return score, reasons


def build_queues(records: list[dict[str, Any]], *, per_role: int) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, int]]]:
    queues: dict[str, list[dict[str, Any]]] = {}
    summary: dict[str, dict[str, int]] = {}
    for role_type in ROLE_TYPES:
        role_rows = [row for row in records if row.get("candidate_role") == role_type and row.get("screening_decision") != "excluded"]
        ranked: list[dict[str, Any]] = []
        for row in role_rows:
            score, reasons = rank_record(row, role_type)
            ranked.append({**row, "review_score": score, "review_evidence": reasons})
        ranked.sort(key=lambda row: (-row["review_score"], not row.get("entry_signal"), row["title"].casefold(), row["id"]))

        company_count: Counter[str] = Counter()
        for position, row in enumerate(ranked, start=1):
            company = row.get("company") or "Unknown"
            company_count[company] += 1
            row["review_rank"] = position
            row["company_rank"] = company_count[company]
            row["recommended_for_review"] = position <= per_role
            row["manual_decision"] = "pending"
            row["manual_notes"] = ""
        queues[role_type] = ranked
        summary[role_type] = {
            "available_candidates": len(ranked),
            "strong_candidates": sum(row["review_score"] >= 6 for row in ranked),
            "recommended_for_review": min(per_role, len(ranked)),
        }
    return queues, summary


def write_queue(path: Path, records: list[dict[str, Any]]) -> None:
    fields = [
        "review_rank", "recommended_for_review", "review_score", "review_evidence", "manual_decision", "manual_notes",
        "id", "source", "title", "company", "location", "published_at", "entry_signal", "role_confidence", "company_rank", "url",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = {field: record.get(field) for field in fields}
            row["review_evidence"] = ";".join(record["review_evidence"])
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create ranked manual-review queues for the Irish corpus.")
    parser.add_argument("--input", type=Path, default=Path("data/clean/irish_screened_candidates.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/irish_review_queues"))
    parser.add_argument("--per-role", type=int, default=50, help="Number of postings to flag for review per role; this is not final selection.")
    args = parser.parse_args()
    records = load_jsonl(args.input)
    queues, summary = build_queues(records, per_role=args.per_role)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for role_type, queue in queues.items():
        write_queue(args.output_dir / f"{role_type}.csv", queue)
    report = {"input": str(args.input), "review_queue_size_per_role": args.per_role, "roles": summary}
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
