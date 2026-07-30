from __future__ import annotations

from types import SimpleNamespace

import src.app as app


def test_build_inspection_rows_formats_candidates(monkeypatch) -> None:
    monkeypatch.setattr(
        app,
        "predict_text_spans",
        lambda bundle, text: [
            SimpleNamespace(text="Python testing", label="SKILL_TECH", confidence=0.91),
            SimpleNamespace(text="teamwork", label="SKILL_TRANS", confidence=0.72),
        ],
    )
    monkeypatch.setattr(
        app,
        "rank_span_candidates",
        lambda span_text, ranker, lexicon_index=None, top_k=3: [
            {
                "label": f"match:{span_text}",
                "method": "exact",
                "score": 1.0,
                "concept_uri": f"uri:{span_text}",
            }
        ],
    )

    rows = app.build_inspection_rows(
        text="Python testing and teamwork.",
        bundle=object(),
        ranker=object(),
        lexicon_index={},
        top_k=3,
    )

    assert len(rows) == 2
    assert rows[0]["span"] == "Python testing"
    assert rows[0]["predicted_uri"] == "uri:Python testing"
    assert "match:Python testing" in rows[0]["top3_candidates"]
    assert rows[1]["label"] == "SKILL_TRANS"
