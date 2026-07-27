from __future__ import annotations

import pandas as pd

from src.extraction.lexicon_baseline import build_lexicon_index, match_span, match_span_detail, normalize_text


def test_normalize_text_strips_punctuation_and_collapses_whitespace() -> None:
    assert normalize_text("  Cloud-Based   Systems, Inc.  ") == "cloud based systems inc"


def test_build_lexicon_index_uses_preferred_and_alt_labels() -> None:
    skills = pd.DataFrame(
        [
            {
                "conceptUri": "http://data.europa.eu/esco/skill/1",
                "preferredLabel": "Cloud support",
                "altLabels": "Cloud infrastructure support\nCloud operations support",
            }
        ]
    )

    index = build_lexicon_index(skills)

    assert index["cloud support"] == "http://data.europa.eu/esco/skill/1"
    assert index["cloud infrastructure support"] == "http://data.europa.eu/esco/skill/1"
    assert index["cloud operations support"] == "http://data.europa.eu/esco/skill/1"


def test_match_span_returns_uri_for_exact_and_fuzzy_matches() -> None:
    skills = pd.DataFrame(
        [
            {
                "conceptUri": "http://data.europa.eu/esco/skill/1",
                "preferredLabel": "Root cause analysis",
                "altLabels": "",
            }
        ]
    )
    index = build_lexicon_index(skills)

    assert match_span("root cause analysis", index) == "http://data.europa.eu/esco/skill/1"
    assert match_span("root cause analisys", index) == "http://data.europa.eu/esco/skill/1"


def test_match_span_detail_reports_method_and_score() -> None:
    skills = pd.DataFrame(
        [
            {
                "conceptUri": "http://data.europa.eu/esco/skill/2",
                "preferredLabel": "Linux",
                "altLabels": "",
            }
        ]
    )
    index = build_lexicon_index(skills)

    result = match_span_detail("Linux", index)

    assert result.method == "exact"
    assert result.score == 100.0
    assert result.concept_uri == "http://data.europa.eu/esco/skill/2"


def test_match_span_rejects_low_overlap_fuzzy_matches() -> None:
    skills = pd.DataFrame(
        [
            {
                "conceptUri": "http://data.europa.eu/esco/skill/3",
                "preferredLabel": "Support preparation of animals for anaesthesia",
                "altLabels": "",
            }
        ]
    )
    index = build_lexicon_index(skills)

    assert match_span("cloud infrastructure support", index) is None
