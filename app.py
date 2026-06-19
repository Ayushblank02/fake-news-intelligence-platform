"""
app.py — Streamlit dashboard for fake news detection.

Features:
  - Text paste mode + URL scrape mode
  - Confidence gate  (UNCERTAIN below threshold)
  - Token-level explainability
  - Prediction history (persisted to CSV)

Usage:
    streamlit run app.py
"""

import json
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from utils import (
    CONFIDENCE_THRESHOLD,
    clean_text,
    combine_features,
    extract_metadata,
    interpret_prediction,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_DIR      = "models/"
HISTORY_FILE   = "prediction_history.csv"
HISTORY_COLS   = ["timestamp", "source", "snippet", "verdict", "confidence"]

TOP_N_EXPLAIN  = 10   # tokens to show in explainability panel


# ---------------------------------------------------------------------------
# Load artifacts (cached — runs once per session)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    vectorizer = joblib.load(os.path.join(MODEL_DIR, "vectorizer.pkl"))
    scaler     = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    model      = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))

    tokens_path = os.path.join(MODEL_DIR, "top_tokens.json")
    with open(tokens_path) as f:
        top_tokens = json.load(f)
    # Rebuild sets for O(1) lookup (JSON doesn't serialise sets)
    top_tokens["fake_set"] = set(top_tokens["fake"])
    top_tokens["real_set"] = set(top_tokens["real"])

    metrics_path = os.path.join(MODEL_DIR, "metrics.json")
    metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)

    return vectorizer, scaler, model, top_tokens, metrics


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def predict(raw_text: str, vectorizer, scaler, model, top_tokens: dict) -> dict:
    """Run full inference pipeline on a single article."""

    # Metadata from raw text (before cleaning — preserves casing/punct signal)
    metadata = extract_metadata([raw_text])

    # TF-IDF on cleaned text
    cleaned = clean_text(raw_text)
    tfidf   = vectorizer.transform([cleaned])
    X       = combine_features(tfidf, metadata, scaler)   # scaler applied here

    label      = int(model.predict(X)[0])
    proba_all  = model.predict_proba(X)[0]
    confidence = float(proba_all[label])

    result = interpret_prediction(label, confidence)

    # Explainability — use vectorizer's own analyzer to generate n-grams
    # This correctly handles bigrams like "breaking news", "white house"
    # that cleaned.split() would never produce as single tokens
    analyze        = vectorizer.build_analyzer()
    article_ngrams = set(analyze(cleaned))

    result["matched_fake_tokens"] = [
        t for t in top_tokens["fake"] if t in article_ngrams
    ][:TOP_N_EXPLAIN]

    result["matched_real_tokens"] = [
        t for t in top_tokens["real"] if t in article_ngrams
    ][:TOP_N_EXPLAIN]

    return result


# ---------------------------------------------------------------------------
# Prediction history
# ---------------------------------------------------------------------------
def load_history() -> pd.DataFrame:
    if os.path.exists(HISTORY_FILE):
        return pd.read_csv(HISTORY_FILE)
    return pd.DataFrame(columns=HISTORY_COLS)


def save_to_history(source: str, raw_text: str, result: dict):
    snippet = raw_text[:120].replace("\n", " ") + ("…" if len(raw_text) > 120 else "")
    row = {
        "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source":     source,
        "snippet":    snippet,
        "verdict":    result["verdict"],
        "confidence": f"{result['confidence']:.1%}",
    }
    df = pd.DataFrame([row])
    df.to_csv(
        HISTORY_FILE,
        mode="a",
        header=not os.path.exists(HISTORY_FILE),
        index=False,
    )


# ---------------------------------------------------------------------------
# URL scraping (optional — works for ~70% of public articles)
# ---------------------------------------------------------------------------
def scrape_url(url: str) -> str | None:
    """
    Extract article text from a URL using newspaper3k.
    Returns None if scraping fails (paywall, JS-heavy site, etc.)
    Install: pip install newspaper3k
    """
    try:
        from newspaper import Article
        article = Article(url)
        article.download()
        article.parse()
        if len(article.text) < 100:
            return None
        return (article.title or "") + "\n\n" + article.text
    except Exception:
        return None


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------
def verdict_color(verdict: str) -> str:
    return {"FAKE": "#D85A30", "REAL": "#1D9E75", "UNCERTAIN": "#BA7517"}[verdict]


def render_verdict(result: dict):
    color = verdict_color(result["verdict"])

    st.markdown(
        f"""
        <div style="
            border: 2px solid {color};
            border-radius: 12px;
            padding: 20px 24px;
            margin: 16px 0;
        ">
            <p style="font-size: 28px; font-weight: 600; color: {color}; margin: 0 0 4px;">
                {result['verdict']}
            </p>
            <p style="font-size: 14px; color: #888; margin: 0 0 12px;">
                Confidence: {result['confidence']:.1%}
            </p>
            <p style="font-size: 14px; margin: 0;">{result['message']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Confidence bar
    st.progress(result["confidence"], text=f"Model confidence: {result['confidence']:.1%}")


def render_explainability(result: dict):
    fake_tokens = result.get("matched_fake_tokens", [])
    real_tokens = result.get("matched_real_tokens", [])

    if not fake_tokens and not real_tokens:
        st.caption("No strongly weighted tokens found in this article.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Signals toward FAKE**")
        if fake_tokens:
            for token in fake_tokens:
                st.markdown(
                    f'<span style="background:#FAECE7; color:#712B13; '
                    f'padding:2px 8px; border-radius:4px; margin:2px; '
                    f'display:inline-block; font-size:13px;">{token}</span>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption("None found")

    with col2:
        st.markdown("**Signals toward REAL**")
        if real_tokens:
            for token in real_tokens:
                st.markdown(
                    f'<span style="background:#E1F5EE; color:#085041; '
                    f'padding:2px 8px; border-radius:4px; margin:2px; '
                    f'display:inline-block; font-size:13px;">{token}</span>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption("None found")

    st.caption(
        "These are words from this article that are strongly associated with "
        "fake or real news in the training data."
    )


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="Fake News Detector",
        page_icon="🔍",
        layout="wide",
    )

    st.title("Fake News Intelligence Platform")
    st.caption(
        "Explainable fake news detection using TF-IDF, feature engineering, "
        "and a calibrated Linear SVM."
    )

    # Load model
    try:
        vectorizer, scaler, model, top_tokens, metrics = load_artifacts()
    except FileNotFoundError:
        st.error(
            "Model artifacts not found. Run `python train.py` first to train the model."
        )
        st.stop()

    # --- Input section ---
    st.subheader("Analyse an article")
    input_mode = st.radio(
        "Input mode",
        ["Paste text", "Enter URL"],
        horizontal=True,
        label_visibility="collapsed",
    )

    raw_text = None
    source_label = "text"

    if input_mode == "Paste text":
        pasted = st.text_area(
            "Paste article text here",
            height=200,
            placeholder="Paste the full article text, including the headline…",
        )
        if pasted.strip():
            raw_text = pasted.strip()
            source_label = "text"

    else:
        url = st.text_input(
            "Article URL",
            placeholder="https://example.com/article",
        )
        if url.strip():
            with st.spinner("Fetching article…"):
                raw_text = scrape_url(url.strip())
            if raw_text:
                st.success(f"Article fetched — {len(raw_text):,} characters")
                source_label = url.strip()
            else:
                st.warning(
                    "Could not extract text from that URL. "
                    "The site may use JavaScript rendering or a paywall. "
                    "Try pasting the article text directly."
                )

    # --- Analyse button ---
    if raw_text:
        if len(raw_text.split()) < 20:
            st.warning("Article is very short. Results may be unreliable.")

        if st.button("Analyse", type="primary", use_container_width=True):
            with st.spinner("Analysing…"):
                result = predict(raw_text, vectorizer, scaler, model, top_tokens)

            render_verdict(result)

            with st.expander("Why this prediction?", expanded=True):
                render_explainability(result)

            save_to_history(source_label, raw_text, result)
            st.toast("Saved to prediction history", icon="✅")

    st.divider()

    # --- Prediction history ---
    st.subheader("Recent analyses")
    history = load_history()

    if history.empty:
        st.caption("No predictions yet. Analyse an article above.")
    else:
        # Colour-code the verdict column
        def highlight_verdict(val):
            colors = {
                "FAKE":      "color: #D85A30; font-weight: 600",
                "REAL":      "color: #1D9E75; font-weight: 600",
                "UNCERTAIN": "color: #BA7517; font-weight: 600",
            }
            return colors.get(val, "")

        display_df = history.iloc[::-1].reset_index(drop=True)  # newest first
        st.dataframe(
            display_df.style.map(highlight_verdict, subset=["verdict"]),
            use_container_width=True,
            hide_index=True,
        )

        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("Clear history"):
                os.remove(HISTORY_FILE)
                st.rerun()

    # --- Sidebar: model info + live metrics ---
    with st.sidebar:
        st.header("Model info")
        st.markdown(
            f"""
            **Architecture**  
            TF-IDF (1–2 grams, 50k features)  
            + 5 scaled metadata features  
            → Calibrated LinearSVC

            **Confidence threshold**  
            `{CONFIDENCE_THRESHOLD:.0%}` — below this, verdict is **UNCERTAIN**

            **Explainability**  
            Token weights from `LinearSVC.coef_`  
            matched using vectorizer n-gram analyzer
            """
        )

        if metrics:
            st.divider()
            st.markdown("**Training metrics**")

            col_a, col_b = st.columns(2)
            col_a.metric("Accuracy",  f"{metrics.get('accuracy', 0):.1%}")
            col_b.metric("ROC-AUC",   f"{metrics.get('roc_auc', 0):.3f}")
            col_a.metric("F1 (macro)", f"{metrics.get('f1_macro', 0):.1%}")

            cv_mean = metrics.get("cv_mean", 0)
            cv_std  = metrics.get("cv_std",  0)
            col_b.metric("CV accuracy", f"{cv_mean:.1%} ± {cv_std:.1%}")

            n_train = metrics.get("n_train", 0)
            n_test  = metrics.get("n_test",  0)
            st.caption(f"Trained on {n_train:,} articles · tested on {n_test:,}")

        st.divider()
        st.caption("Fake News Intelligence Platform · v1.0")


if __name__ == "__main__":
    main()
