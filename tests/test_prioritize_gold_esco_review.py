from __future__ import annotations

from src.evaluation.prioritize_gold_esco_review import prioritize_review_rows


def test_prioritize_review_rows_puts_uncertain_spans_first() -> None:
    rows = [
        {
            "role_type": "software_dev",
            "posting_id": "1",
            "span_index": "1",
            "lexicon_uri": "uri-lex",
            "candidate_1_uri": "uri-lex",
            "candidate_1_score": "1.0",
            "candidate_2_score": "0.1",
        },
        {
            "role_type": "software_dev",
            "posting_id": "2",
            "span_index": "1",
            "lexicon_uri": "",
            "candidate_1_uri": "uri-top1",
            "candidate_1_score": "0.5",
            "candidate_2_score": "0.48",
        },
    ]

    prioritized = prioritize_review_rows(rows)
    assert prioritized[0]["provisional_source"] == "embedding_top1"
    assert prioritized[0]["needs_review"] == "yes"
    assert prioritized[1]["provisional_source"] == "lexicon_exact"
