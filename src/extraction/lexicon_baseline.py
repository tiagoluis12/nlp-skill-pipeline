from __future__ import annotations

import argparse
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
from rapidfuzz import fuzz, process


_NON_ALNUM_RE = re.compile(r"[^0-9a-zA-Z\s]+")
_WHITESPACE_RE = re.compile(r"\s+")
_ALT_LABEL_SPLIT_RE = re.compile(r"[\n;|]+")


@dataclass(frozen=True)
class MatchResult:
    query: str
    normalized_query: str
    matched_label: str | None
    concept_uri: str | None
    score: float
    method: str


def normalize_text(text: str) -> str:
    """Lowercase and remove punctuation while preserving word boundaries."""
    normalized = unicodedata.normalize("NFKC", str(text)).lower()
    normalized = _NON_ALNUM_RE.sub(" ", normalized)
    normalized = _WHITESPACE_RE.sub(" ", normalized).strip()
    return normalized


def _token_set(text: str) -> set[str]:
    return {token for token in normalize_text(text).split() if token}


def _min_shared_tokens(query: str) -> int:
    token_count = len(_token_set(query))
    if token_count <= 1:
        return 1
    return max(2, (token_count + 1) // 2)


def _is_missing(value: object) -> bool:
    return value is None or (isinstance(value, float) and pd.isna(value))


def iter_alt_labels(value: object) -> Iterable[str]:
    if _is_missing(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    return [part.strip() for part in _ALT_LABEL_SPLIT_RE.split(text) if part.strip()]


def load_esco_skills(csv_path: Path) -> pd.DataFrame:
    skills = pd.read_csv(csv_path)
    required_columns = {"conceptUri", "preferredLabel"}
    missing = sorted(required_columns - set(skills.columns))
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {', '.join(missing)}")
    return skills


def build_lexicon_index(skills: pd.DataFrame) -> dict[str, str]:
    index: dict[str, str] = {}

    for _, row in skills.iterrows():
        concept_uri = str(row["conceptUri"]).strip()
        if not concept_uri:
            continue

        labels = [row.get("preferredLabel")]
        if "altLabels" in skills.columns:
            labels.extend(iter_alt_labels(row.get("altLabels")))

        for label in labels:
            if _is_missing(label):
                continue
            normalized = normalize_text(label)
            if not normalized:
                continue
            index.setdefault(normalized, concept_uri)

    return index


def build_token_index(index: dict[str, str]) -> dict[str, set[str]]:
    token_index: dict[str, set[str]] = {}
    for label in index:
        for token in _token_set(label):
            token_index.setdefault(token, set()).add(label)
    return token_index


def candidate_labels_for_span(
    span: str,
    index: dict[str, str],
    token_index: dict[str, set[str]] | None = None,
    max_candidates: int = 2500,
) -> list[str]:
    normalized = normalize_text(span)
    if normalized in index:
        return [normalized]

    if token_index is None:
        return list(index.keys())

    tokens = [token for token in normalized.split() if token]
    if not tokens:
        return []

    candidate_sets = [token_index.get(token, set()) for token in tokens]
    candidate_sets = [candidates for candidates in candidate_sets if candidates]
    if not candidate_sets:
        return []

    if len(candidate_sets) == 1:
        candidates = set(candidate_sets[0])
    else:
        candidates = set().union(*candidate_sets)

    if len(candidates) > max_candidates:
        return []

    return sorted(candidates)


def match_span(
    span: str,
    index: dict[str, str],
    score_cutoff: int = 85,
    candidate_labels: Sequence[str] | None = None,
) -> str | None:
    normalized = normalize_text(span)
    if normalized in index:
        return index[normalized]

    choices = candidate_labels if candidate_labels is not None else index.keys()
    result = process.extractOne(
        normalized,
        choices,
        scorer=fuzz.WRatio,
        score_cutoff=score_cutoff,
    )
    if result is None:
        return None

    matched_label = result[0]
    if len(_token_set(span) & _token_set(matched_label)) < _min_shared_tokens(span):
        return None
    return index[matched_label]


def match_span_detail(
    span: str,
    index: dict[str, str],
    score_cutoff: int = 85,
    candidate_labels: Sequence[str] | None = None,
) -> MatchResult:
    normalized = normalize_text(span)
    if normalized in index:
        return MatchResult(
            query=span,
            normalized_query=normalized,
            matched_label=normalized,
            concept_uri=index[normalized],
            score=100.0,
            method="exact",
        )

    choices = candidate_labels if candidate_labels is not None else index.keys()
    result = process.extractOne(
        normalized,
        choices,
        scorer=fuzz.WRatio,
        score_cutoff=score_cutoff,
    )
    if result is None:
        return MatchResult(
            query=span,
            normalized_query=normalized,
            matched_label=None,
            concept_uri=None,
            score=0.0,
            method="none",
        )

    matched_label, score, _ = result
    if len(_token_set(span) & _token_set(matched_label)) < _min_shared_tokens(span):
        return MatchResult(
            query=span,
            normalized_query=normalized,
            matched_label=None,
            concept_uri=None,
            score=0.0,
            method="none",
        )

    return MatchResult(
        query=span,
        normalized_query=normalized,
        matched_label=matched_label,
        concept_uri=index[matched_label],
        score=float(score),
        method="fuzzy" if normalized != matched_label else "exact",
    )


def match_spans(spans: Sequence[str], index: dict[str, str], score_cutoff: int = 85) -> list[MatchResult]:
    return [match_span_detail(span, index, score_cutoff=score_cutoff) for span in spans]


def summarize_index(skills: pd.DataFrame, index: dict[str, str]) -> dict[str, int]:
    preferred_labels = 0
    alt_labels = 0
    for _, row in skills.iterrows():
        if str(row.get("preferredLabel", "")).strip():
            preferred_labels += 1
        if "altLabels" in skills.columns:
            alt_labels += sum(1 for label in iter_alt_labels(row.get("altLabels")))

    return {
        "rows": int(len(skills)),
        "preferred_labels": preferred_labels,
        "alt_labels": alt_labels,
        "index_size": len(index),
    }


def write_index(index: dict[str, str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and query the ESCO lexicon baseline.")
    parser.add_argument("--skills-csv", type=Path, required=True, help="Path to ESCO skills_en.csv")
    parser.add_argument("--output-index", type=Path, help="Optional JSON output path for the normalized index")
    parser.add_argument("--query", action="append", default=[], help="Skill span to match; can be passed multiple times")
    parser.add_argument("--score-cutoff", type=int, default=85)
    args = parser.parse_args()

    skills = load_esco_skills(args.skills_csv)
    index = build_lexicon_index(skills)
    summary = summarize_index(skills, index)

    payload: dict[str, object] = {"summary": summary, "matches": []}
    if args.query:
        payload["matches"] = [match_span_detail(query, index, score_cutoff=args.score_cutoff).__dict__ for query in args.query]

    if args.output_index is not None:
        write_index(index, args.output_index)

    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
