from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import pandas as pd
import streamlit as st

from src.esco.ranker import load_lexicon_index, load_span_ranker, rank_span_candidates
from src.extraction.ner_model import (
    SpanPrediction,
    load_checkpoint_bundle,
    predict_text_spans,
)


ESCO_CSV = Path("data/benchmarks/esco/skills_en.csv")
ESCO_EMBEDDINGS = Path("models/esco_embeddings.npy")
ESCO_INDEX = Path("models/esco_faiss.index")
ESCO_LABELS = Path("models/esco_embedding_labels.json")


def _candidate_summary(candidates: Sequence[dict[str, object]]) -> str:
    lines: list[str] = []
    for candidate in candidates[:3]:
        label = str(candidate.get("label") or "")
        method = str(candidate.get("method") or "")
        score = float(candidate.get("score") or 0.0)
        uri = str(candidate.get("concept_uri") or "")
        lines.append(f"{label} [{method}] {score:.2f} {uri}")
    return "\n".join(lines)


def _build_sentence_context(text: str, span: SpanPrediction) -> str:
    start = max(0, text.lower().find(span.text.lower())) if span.text else -1
    if start < 0:
        return text.strip()
    prefix = text[:start].rsplit(".", 1)[-1].rsplit("!", 1)[-1].rsplit("?", 1)[-1]
    suffix = text[start + len(span.text) :]
    sentence = (prefix + span.text + suffix).strip()
    return sentence or text.strip()


def build_inspection_rows(
    text: str,
    bundle: object,
    ranker: object,
    lexicon_index: dict[str, str],
    top_k: int = 3,
) -> list[dict[str, object]]:
    spans = predict_text_spans(bundle, text)
    rows: list[dict[str, object]] = []
    for rank, span in enumerate(spans, start=1):
        candidates = rank_span_candidates(span.text, ranker, lexicon_index=lexicon_index, top_k=top_k)
        top_candidate = candidates[0] if candidates else {}
        rows.append(
            {
                "rank": rank,
                "span": span.text,
                "label": span.label,
                "confidence": round(float(span.confidence), 3),
                "predicted_uri": top_candidate.get("concept_uri"),
                "top_method": top_candidate.get("method"),
                "top_score": round(float(top_candidate.get("score") or 0.0), 3) if top_candidate else 0.0,
                "top3_candidates": _candidate_summary(candidates),
                "source_sentence": _build_sentence_context(text, span),
            }
        )
    return rows


@st.cache_resource(show_spinner=False)
def load_demo_resources() -> dict[str, object]:
    bundle = load_checkpoint_bundle()
    ranker = load_span_ranker(ESCO_EMBEDDINGS, ESCO_INDEX, ESCO_LABELS)
    lexicon_index = load_lexicon_index(ESCO_CSV)
    return {
        "bundle": bundle,
        "ranker": ranker,
        "lexicon_index": lexicon_index,
    }


def render_app() -> None:
    st.set_page_config(
        page_title="Skill Extraction Pipeline",
        page_icon=":mag:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #f7f3ec 0%, #ffffff 45%, #f2f6f8 100%);
        }
        .hero {
            padding: 1.4rem 1.5rem;
            border-radius: 1rem;
            background: rgba(16, 24, 40, 0.92);
            color: white;
            margin-bottom: 1rem;
        }
        .hero h1 {
            margin: 0;
            font-size: 2rem;
        }
        .hero p {
            margin: 0.35rem 0 0;
            opacity: 0.88;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="hero">
          <h1>Skill Extraction Pipeline</h1>
          <p>Phase 4 demo: NER span detection, ESCO ranking, and inspection-friendly outputs.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.subheader("Controls")
        top_k = st.slider("ESCO candidates", min_value=1, max_value=5, value=3)
        show_raw = st.checkbox("Show raw trace JSON", value=False)
        st.caption("The app uses the cached Phase 4 checkpoint and the ESCO embedding index.")

    text = st.text_area("Paste job posting text", height=240, placeholder="Paste a job posting or a paragraph here.")
    extract_clicked = st.button("Inspect Posting")

    if not extract_clicked:
        st.info("Paste text and click Inspect Posting.")
        return

    if not text.strip():
        st.warning("Paste some text first.")
        return

    resources = load_demo_resources()
    rows = build_inspection_rows(
        text=text,
        bundle=resources["bundle"],
        ranker=resources["ranker"],
        lexicon_index=resources["lexicon_index"],
        top_k=top_k,
    )

    st.metric("Detected spans", len(rows))
    if not rows:
        st.info("No skill spans were detected.")
        return

    display_df = pd.DataFrame(rows)[
        ["rank", "span", "label", "confidence", "predicted_uri", "top_method", "top_score", "source_sentence"]
    ]
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.subheader("Candidate Detail")
    for row in rows:
        with st.expander(f"{row['rank']}. {row['span']} ({row['label']})", expanded=row["rank"] == 1):
            left, right = st.columns([2, 1])
            with left:
                st.write(f"Confidence: `{row['confidence']}`")
                st.write(f"Predicted URI: `{row['predicted_uri']}`")
                st.write(f"Top method: `{row['top_method']}`")
                st.write("Source sentence:")
                st.write(row["source_sentence"])
            with right:
                st.text_area(
                    "Top candidates",
                    value=row["top3_candidates"],
                    height=160,
                    key=f"cand-{row['rank']}",
                )

    if show_raw:
        st.subheader("Raw Trace")
        st.code(json.dumps(rows, indent=2, sort_keys=True, ensure_ascii=False), language="json")


def main() -> None:
    render_app()


if __name__ == "__main__":
    main()
