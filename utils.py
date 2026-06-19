"""
utils.py — shared preprocessing for training AND inference.

Both train.py and app.py import from here.
This guarantees training/inference parity: the exact same
transformations are applied at both stages.
"""

import re
import string
import numpy as np
import scipy.sparse as sp
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ---------------------------------------------------------------------------
# Initialise once at import time (not inside the function — this is expensive)
# ---------------------------------------------------------------------------
_lemmatizer = WordNetLemmatizer()
_stopwords = set(stopwords.words("english"))

# Contractions to expand before cleaning
_CONTRACTIONS = {
    "i'm": "i am", "he's": "he is", "she's": "she is", "that's": "that is",
    "what's": "what is", "where's": "where is", "it's": "it is",
    "'ll": " will", "'ve": " have", "'re": " are", "'d": " would",
    "won't": "will not", "can't": "cannot", "n't": " not",
}

CONFIDENCE_THRESHOLD = 0.65   # below this → "Uncertain"


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """
    Normalise raw article text for TF-IDF vectorisation.

    Steps (order matters):
      1. Lower-case
      2. Strip URLs and HTML tags
      3. Expand contractions
      4. Remove punctuation and digits
      5. Tokenise, remove stopwords, lemmatise
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    text = text.lower()

    # Remove URLs (http/https and bare www.)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Expand contractions
    for contraction, expansion in _CONTRACTIONS.items():
        text = text.replace(contraction, expansion)

    # Remove punctuation and digits — keep plain alphabetic tokens
    text = re.sub(r"[^a-z\s]", " ", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenise → remove stopwords → lemmatise
    tokens = [
        _lemmatizer.lemmatize(token)
        for token in text.split()
        if token not in _stopwords and len(token) > 2
    ]

    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Metadata features
# ---------------------------------------------------------------------------
def extract_metadata(texts) -> np.ndarray:
    """
    Compute 5 hand-crafted features from raw (uncleaned) text.

    Uses the ORIGINAL text, not the cleaned version — punctuation
    and casing carry signal that cleaning intentionally removes.

    Returns shape (n_samples, 5):
      0: text_length       — total character count
      1: word_count        — whitespace-separated tokens
      2: avg_word_length   — mean chars per word
      3: punct_ratio       — punctuation chars / total chars
      4: uppercase_ratio   — uppercase chars / total alpha chars
    """
    rows = []
    for text in texts:
        if not isinstance(text, str) or not text.strip():
            rows.append([0, 0, 0.0, 0.0, 0.0])
            continue

        words = text.split()
        n_chars = len(text)
        n_words = len(words)
        avg_word_len = np.mean([len(w) for w in words]) if words else 0.0
        n_punct = sum(1 for c in text if c in string.punctuation)
        punct_ratio = n_punct / n_chars if n_chars > 0 else 0.0
        alpha_chars = [c for c in text if c.isalpha()]
        upper_ratio = (
            sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
            if alpha_chars else 0.0
        )

        rows.append([n_chars, n_words, avg_word_len, punct_ratio, upper_ratio])

    return np.array(rows, dtype=np.float32)


# ---------------------------------------------------------------------------
# Feature combination
# ---------------------------------------------------------------------------
def combine_features(tfidf_matrix, metadata_array,
                     scaler=None) -> sp.csr_matrix:
    """
    Scale metadata, then horizontally stack with TF-IDF sparse matrix.

    Why scale metadata separately?
    TF-IDF values are already small floats (0–1 range after sublinear_tf).
    Raw metadata like text_length can be 5000+ — without scaling it would
    dominate the SVM margin calculation and drown out the text signal.

    Args:
        tfidf_matrix:   sparse matrix (n_samples, n_tfidf_features)
        metadata_array: dense array  (n_samples, 5)
        scaler:         fitted StandardScaler, or None (no scaling applied).
                        Pass the scaler fitted on training data at inference
                        time — never refit on the inference sample.
    """
    if scaler is not None:
        metadata_array = scaler.transform(metadata_array)
    meta_sparse = sp.csr_matrix(metadata_array)
    return sp.hstack([tfidf_matrix, meta_sparse], format="csr")


# ---------------------------------------------------------------------------
# Prediction interpretation
# ---------------------------------------------------------------------------
def interpret_prediction(label: int, proba: float) -> dict:
    """
    Convert raw model output into a display-ready result dict.

    Args:
        label:  0 = real, 1 = fake
        proba:  confidence for the predicted class (0.0 – 1.0)

    Returns dict with keys:
        verdict:     "FAKE" | "REAL" | "UNCERTAIN"
        confidence:  float 0–1
        label_int:   original int label
        message:     human-readable explanation
    """
    if proba < CONFIDENCE_THRESHOLD:
        return {
            "verdict": "UNCERTAIN",
            "confidence": proba,
            "label_int": label,
            "message": (
                "This article contains mixed signals. "
                "Manual verification is recommended."
            ),
        }

    verdict = "FAKE" if label == 1 else "REAL"
    message = (
        "High indicators of misleading or fabricated content detected."
        if verdict == "FAKE"
        else "No strong indicators of misinformation detected."
    )

    return {
        "verdict": verdict,
        "confidence": proba,
        "label_int": label,
        "message": message,
    }
