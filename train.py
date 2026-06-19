"""
train.py — full training pipeline.

Run once to produce:
    models/vectorizer.pkl
    models/scaler.pkl
    models/model.pkl
    models/top_tokens.json
    models/metrics.json

Usage:
    python train.py --data_dir data/ --model_dir models/
"""

import argparse
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from utils import clean_text, combine_features, extract_metadata

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TFIDF_PARAMS = {
    "ngram_range": (1, 2),   # unigrams + bigrams
    "max_features": 50_000,
    "sublinear_tf": True,    # log(1 + tf) — compresses high-frequency terms
    "min_df": 2,             # ignore terms that appear in only 1 document
    "strip_accents": "unicode",
    "analyzer": "word",
}

SVC_PARAMS = {
    "C": 1.0,
    "max_iter": 2000,        # enough iterations to guarantee convergence
    "dual": "auto",
}

TOP_N_TOKENS = 20            # tokens to save for explainability


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(data_dir: str) -> pd.DataFrame:
    """
    Load and merge Fake.csv + True.csv.
    Expects columns: title, text, subject, date.
    Label: 1 = fake, 0 = real.
    """
    fake_path = os.path.join(data_dir, "Fake.csv")
    real_path = os.path.join(data_dir, "True.csv")

    fake = pd.read_csv(fake_path)
    real = pd.read_csv(real_path)

    fake["label"] = 1
    real["label"] = 0

    df = pd.concat([fake, real], ignore_index=True)

    # Combine title + body — title carries strong signal
    df["raw_text"] = (
        df["title"].fillna("") + " " + df["text"].fillna("")
    ).str.strip()

    # Drop rows with no usable text
    df = df[df["raw_text"].str.len() > 20].copy()

    print(f"  Loaded {len(df):,} articles  "
          f"({fake['label'].count():,} fake / {real['label'].count():,} real)")

    return df


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def preprocess(df: pd.DataFrame):
    """Clean text and extract metadata. Returns parallel arrays."""
    print("  Cleaning text …", end=" ", flush=True)
    t0 = time.time()
    df["clean"] = df["raw_text"].apply(clean_text)
    print(f"done ({time.time() - t0:.1f}s)")

    print("  Extracting metadata features …", end=" ", flush=True)
    metadata = extract_metadata(df["raw_text"].tolist())
    print("done")

    return df["clean"].tolist(), metadata, df["label"].tolist()


# ---------------------------------------------------------------------------
# Explainability helper
# ---------------------------------------------------------------------------
def extract_top_tokens(vectorizer: TfidfVectorizer,
                       calibrated_model: CalibratedClassifierCV,
                       n: int = TOP_N_TOKENS) -> dict:
    """
    Extract the n tokens most associated with FAKE and REAL predictions.

    Uses the raw LinearSVC coefficients from inside the calibrated wrapper.
    Positive coef → pushes toward class 1 (FAKE).
    Negative coef → pushes toward class 0 (REAL).
    """
    coefs = np.mean(
        [clf.estimator.coef_[0] for clf in calibrated_model.calibrated_classifiers_],
        axis=0,
    )

    feature_names = np.array(vectorizer.get_feature_names_out())

    # Only consider TF-IDF features (metadata appended at the end)
    n_tfidf = len(feature_names)
    coefs_tfidf = coefs[:n_tfidf]

    top_fake_idx = np.argsort(coefs_tfidf)[-n:][::-1]
    top_real_idx = np.argsort(coefs_tfidf)[:n]

    return {
    "fake": feature_names[top_fake_idx].tolist(),
    "real": feature_names[top_real_idx].tolist(),
}    


def get_article_ngrams(cleaned_text: str, vectorizer: TfidfVectorizer) -> set:
    """
    Generate all n-grams from a cleaned article using the vectorizer's
    own analyzer — this handles ngram_range, tokenizer, and preprocessor
    identically to how the vectorizer saw the training data.

    Using cleaned.split() would miss bigrams like 'breaking news', 'white house',
    'donald trump' — which are often the highest-weighted explainability tokens.
    """
    analyze = vectorizer.build_analyzer()
    return set(analyze(cleaned_text))


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def train(data_dir: str, model_dir: str):
    os.makedirs(model_dir, exist_ok=True)

    print("\n[1/6] Loading data …")
    df = load_data(data_dir)

    print("\n[2/6] Preprocessing …")
    texts_clean, metadata, labels = preprocess(df)

    print("\n[3/6] Splitting …")
    (
        X_text_train, X_text_test,
        X_meta_train, X_meta_test,
        y_train, y_test,
    ) = train_test_split(
        texts_clean, metadata, labels,
        test_size=0.20,
        random_state=42,
        stratify=labels,
    )
    print(f"  Train: {len(y_train):,}   Test: {len(y_test):,}")

    print("\n[4/6] Fitting TF-IDF + scaler + SVM …")
    t0 = time.time()

    # TF-IDF on text
    vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
    X_tfidf_train = vectorizer.fit_transform(X_text_train)

    # Scale metadata on train split only — never fit on test data
    scaler = StandardScaler()
    scaler.fit(X_meta_train)                         # fit on train
    X_train = combine_features(X_tfidf_train, X_meta_train, scaler)

    svc = LinearSVC(**SVC_PARAMS)
    model = CalibratedClassifierCV(svc, cv=5)
    model.fit(X_train, y_train)
    print(f"  Fit complete ({time.time() - t0:.1f}s)")

    print("\n[5/6] Evaluating …")
    X_tfidf_test = vectorizer.transform(X_text_test)
    X_test = combine_features(X_tfidf_test, X_meta_test, scaler)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    f1       = f1_score(y_test, y_pred, average="macro")
    roc_auc  = roc_auc_score(y_test, y_proba)

    print("\n" + classification_report(y_test, y_pred,
                                       target_names=["Real", "Fake"]))
    print("Confusion matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print(f"\nROC-AUC: {roc_auc:.4f}")

    # Cross-validation on full training set — proves robustness across folds
    # Uses the same vectorizer features; quick because SVM is fast
    print("\n  Running 5-fold cross-validation …", end=" ", flush=True)
    X_tfidf_full = vectorizer.transform(texts_clean)
    meta_full    = extract_metadata(df["raw_text"].tolist())
    X_full       = combine_features(X_tfidf_full, meta_full, scaler)
    cv_scores    = cross_val_score(model, X_full, labels, cv=5, scoring="accuracy", n_jobs=-1)
    print("done")
    print(f"  CV Accuracy: {cv_scores.mean():.1%} ± {cv_scores.std():.1%}")

    print("\n[6/6] Saving artifacts …")

    # Build confusion matrix as a serialisable list for the dashboard
    cm_list = cm.tolist()

    metrics = {
        "accuracy":    round(float(accuracy), 4),
        "f1_macro":    round(float(f1), 4),
        "roc_auc":     round(float(roc_auc), 4),
        "cv_mean":     round(float(cv_scores.mean()), 4),
        "cv_std":      round(float(cv_scores.std()), 4),
        "confusion_matrix": cm_list,        # [[TN, FP], [FN, TP]]
        "n_train":     len(y_train),
        "n_test":      len(y_test),
    }

    top_tokens = extract_top_tokens(vectorizer, model)

    paths = {
        "vectorizer": os.path.join(model_dir, "vectorizer.pkl"),
        "scaler":     os.path.join(model_dir, "scaler.pkl"),
        "model":      os.path.join(model_dir, "model.pkl"),
        "tokens":     os.path.join(model_dir, "top_tokens.json"),
        "metrics":    os.path.join(model_dir, "metrics.json"),
    }

    joblib.dump(vectorizer, paths["vectorizer"])
    joblib.dump(scaler,     paths["scaler"])
    joblib.dump(model,      paths["model"])

    with open(paths["tokens"], "w") as f:
        json.dump(top_tokens, f, indent=2)
    with open(paths["metrics"], "w") as f:
        json.dump(metrics, f, indent=2)

    for name, path in paths.items():
        print(f"  {path}")

    print(f"\nTop FAKE tokens: {top_tokens['fake'][:10]}")
    print(f"Top REAL tokens: {top_tokens['real'][:10]}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train fake news classifier")
    parser.add_argument("--data_dir",  default="data/",   help="Directory with Fake.csv and True.csv")
    parser.add_argument("--model_dir", default="models/", help="Output directory for artifacts")
    args = parser.parse_args()

    train(args.data_dir, args.model_dir)
