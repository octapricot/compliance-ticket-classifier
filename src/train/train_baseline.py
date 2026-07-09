"""
Baseline classifier: TF-IDF + Logistic Regression.

Trains on train.parquet, evaluates on the REAL test.parquet.
Logs metrics and the model to Weights & Biases.

The baseline exists to give a number the transformer must beat, to validate
the data pipeline end-to-end, and to provide an interpretable comparison.

Run from project root: python src/train/train_baseline.py
"""
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, accuracy_score,
)

import wandb

load_dotenv()  # loads WANDB_API_KEY

TRAIN = Path("data/labeled/train.parquet")
TEST = Path("data/labeled/test.parquet")

# We treat "relevant" as the positive class (label 1).
POS_LABEL = "relevant"


def main():
    train_df = pd.read_parquet(TRAIN)
    test_df = pd.read_parquet(TEST)
    print(f"Train: {len(train_df)} | Test: {len(test_df)}")

    # Binary target: 1 = relevant, 0 = not_relevant
    y_train = (train_df["label"] == POS_LABEL).astype(int)
    y_test = (test_df["label"] == POS_LABEL).astype(int)
    X_train = train_df["text"]
    X_test = test_df["text"]

    # --- Start W&B run ---
    run = wandb.init(
        project="compliance-ticket-classifier",
        name="baseline-tfidf-logreg",
        config={
            "model": "tfidf+logreg",
            "tfidf_max_features": 5000,
            "tfidf_ngram_range": "1-2",
            "class_weight": "balanced",
            "train_size": len(train_df),
            "test_size": len(test_df),
            "train_synthetic": int(train_df["is_synthetic"].sum()),
        },
    )

    # --- Build & train the pipeline ---
    # class_weight="balanced" helps with our imbalanced classes.
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),   # unigrams + bigrams
            min_df=2,             # ignore words in <2 docs (noise)
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
        )),
    ])
    pipe.fit(X_train, y_train)

    # --- Evaluate on REAL test set ---
    y_pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print("\n=== Test performance (real data only) ===")
    print(classification_report(y_test, y_pred,
          target_names=["not_relevant", "relevant"], zero_division=0))
    print("Confusion matrix [rows=true, cols=pred]:")
    print(cm)

    # --- Log to W&B ---
    wandb.log({
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
    })
    # Confusion matrix as a W&B plot
    wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(
        preds=y_pred.tolist(), y_true=y_test.tolist(),
        class_names=["not_relevant", "relevant"],
    )})

    # Show the most informative words (interpretability!)
    vec = pipe.named_steps["tfidf"]
    clf = pipe.named_steps["clf"]
    feature_names = vec.get_feature_names_out()
    coefs = clf.coef_[0]
    top_pos = sorted(zip(coefs, feature_names), reverse=True)[:15]
    top_neg = sorted(zip(coefs, feature_names))[:15]
    print("\nTop words -> RELEVANT:")
    for c, w in top_pos:
        print(f"  {c:+.2f}  {w}")
    print("\nTop words -> NOT relevant:")
    for c, w in top_neg:
        print(f"  {c:+.2f}  {w}")

    run.finish()
    print("\nLogged to W&B. Baseline done.")


if __name__ == "__main__":
    main()
