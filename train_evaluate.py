
from pathlib import Path
import json
import re
import time
import warnings

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "bias_clean.csv"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"

RESULTS.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)

SEED = 42
LABELS = ["left", "center", "right"]


def normalize_label(value):
    """Map the original five labels to three political-leaning classes."""
    label = str(value).strip().lower().replace("_", "-")

    mapping = {
        "left": "left",
        "leaning-left": "left",
        "center": "center",
        "centre": "center",
        "right": "right",
        "leaning-right": "right",
    }
    return mapping.get(label)


def clean_article(text, site=None):
    """Basic text cleaning and removal of exact outlet-name mentions."""
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b", " ", text)

    # Remove explicit outlet names, but not generic political vocabulary.
    # This is a heuristic, not complete source-leakage elimination.
    if site and str(site).strip().lower() not in {"", "nan", "none"}:
        outlet = re.escape(str(site).strip().lower())
        text = re.sub(rf"(?<!\w){outlet}(?!\w)", " ", text)

    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {DATA_PATH}. Place bias_clean.csv in the repo root."
        )

    df = pd.read_csv(DATA_PATH)

    required = {"page_text", "bias", "site"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    print("\nDataset shape:", df.shape)
    print("\nOriginal labels:")
    print(df["bias"].value_counts(dropna=False))
    print("\nUnique outlets:", df["site"].nunique())

    df["label"] = df["bias"].apply(normalize_label)
    df = df.dropna(subset=["label", "page_text", "site"]).copy()
    df["page_text"] = df["page_text"].fillna("")
    df["site"] = df["site"].astype(str)

    df["text"] = [
        clean_article(text, site)
        for text, site in zip(df["page_text"], df["site"])
    ]

    df = df[df["text"].str.len() >= 50].copy()

    # Remove exact duplicate article texts before splitting.
    before = len(df)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    print(f"\nRemoved {before - len(df)} duplicate texts.")
    print("Usable rows:", len(df))
    print("\nNormalized labels:")
    print(df["label"].value_counts())

    if set(df["label"].unique()) != set(LABELS):
        raise ValueError(
            "Expected all three labels: left, center, right. "
            "Inspect the normalized label distribution."
        )

    return df


def make_splits(df):
    indices = list(range(len(df)))
    y = df["label"]
    groups = df["site"]

    # Random split: stratify to preserve the label proportions.
    random_train, random_test = train_test_split(
        indices,
        test_size=0.20,
        random_state=SEED,
        stratify=y,
    )

    # Media split: keep each outlet entirely in one partition.
    # Try multiple candidate splits to avoid a test partition missing a class.
    media_train, media_test = None, None
    splitter = GroupShuffleSplit(
        n_splits=100,
        test_size=0.20,
        random_state=SEED,
    )

    for train_idx, test_idx in splitter.split(
        df, y, groups=groups
    ):
        train_labels = set(y.iloc[train_idx])
        test_labels = set(y.iloc[test_idx])

        if train_labels == set(LABELS) and test_labels == set(LABELS):
            media_train, media_test = train_idx, test_idx
            break

    if media_train is None:
        raise RuntimeError(
            "Could not find an outlet-disjoint split containing all classes. "
            "Inspect label counts by site and adjust the group split."
        )

    # Save the exact row indices and split type for reproducibility.
    split_dir = RESULTS / "splits"
    split_dir.mkdir(exist_ok=True)

    for name, train_idx, test_idx in [
        ("random", random_train, random_test),
        ("media", media_train, media_test),
    ]:
        pd.DataFrame({
            "row_index": train_idx,
            "partition": "train",
        }).to_csv(split_dir / f"{name}_train.csv", index=False)

        pd.DataFrame({
            "row_index": test_idx,
            "partition": "test",
        }).to_csv(split_dir / f"{name}_test.csv", index=False)

        train_sites = set(groups.iloc[train_idx])
        test_sites = set(groups.iloc[test_idx])

        print(f"\n{name.upper()} SPLIT")
        print("Train articles:", len(train_idx))
        print("Test articles:", len(test_idx))
        print("Train outlets:", len(train_sites))
        print("Test outlets:", len(test_sites))
        print("Outlet overlap:", len(train_sites & test_sites))
        print("\nTest label counts:")
        print(y.iloc[test_idx].value_counts())

        if name == "media" and train_sites & test_sites:
            raise AssertionError("Media split contains outlet leakage.")

    return {
        "random": (random_train, random_test),
        "media": (media_train, media_test),
    }


def build_model(model_name):
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_features=150_000,
        sublinear_tf=True,
        dtype="float32",
    )

    if model_name == "MultinomialNB":
        classifier = MultinomialNB(alpha=0.5)
    elif model_name == "LinearSVC":
        classifier = LinearSVC(
            C=1.0,
            class_weight="balanced",
            random_state=SEED,
            max_iter=5000,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return Pipeline([
        ("tfidf", vectorizer),
        ("classifier", classifier),
    ])


def evaluate(df, split_name, train_idx, test_idx, model_name):
    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

    X_train = train_df["text"]
    y_train = train_df["label"]
    X_test = test_df["text"]
    y_test = test_df["label"]

    model = build_model(model_name)

    start = time.perf_counter()
    model.fit(X_train, y_train)
    train_seconds = time.perf_counter() - start

    start = time.perf_counter()
    predictions = model.predict(X_test)
    predict_seconds = time.perf_counter() - start

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(
        y_test, predictions, labels=LABELS,
        average="macro", zero_division=0
    )
    weighted_f1 = f1_score(
        y_test, predictions, average="weighted", zero_division=0
    )
    macro_precision = precision_score(
        y_test, predictions, labels=LABELS,
        average="macro", zero_division=0
    )
    macro_recall = recall_score(
        y_test, predictions, labels=LABELS,
        average="macro", zero_division=0
    )

    report = classification_report(
        y_test,
        predictions,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )

    run_name = f"{split_name}_{model_name.lower()}"
    model_path = MODELS / f"{run_name}.joblib"
    joblib.dump(model, model_path)

    pd.DataFrame(report).T.to_csv(
        RESULTS / f"{run_name}_classification_report.csv"
    )

    cm = confusion_matrix(y_test, predictions, labels=LABELS)
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LABELS,
        yticklabels=LABELS,
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(f"Confusion Matrix: {run_name}")
    plt.tight_layout()
    plt.savefig(RESULTS / f"{run_name}_confusion_matrix.png", dpi=160)
    plt.close()

    # Save article-level predictions for error analysis.
    output = test_df[["url", "title", "site", "bias", "label"]].copy()
    output["predicted_label"] = predictions
    output["correct"] = output["label"] == output["predicted_label"]
    output.to_csv(RESULTS / f"{run_name}_predictions.csv", index=False)

    metrics = {
        "split": split_name,
        "model": model_name,
        "train_size": len(train_df),
        "test_size": len(test_df),
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "train_seconds": train_seconds,
        "predict_seconds": predict_seconds,
        "model_path": str(model_path),
    }

    print(f"\n{'=' * 55}")
    print(f"{run_name}")
    print(f"{'=' * 55}")
    print(json.dumps(metrics, indent=2))
    print("\nPer-class report:")
    print(classification_report(
        y_test, predictions, labels=LABELS, zero_division=0
    ))

    return metrics


def main():
    df = load_data()
    splits = make_splits(df)

    all_metrics = []

    for split_name, (train_idx, test_idx) in splits.items():
        for model_name in ["MultinomialNB", "LinearSVC"]:
            metrics = evaluate(
                df, split_name, train_idx, test_idx, model_name
            )
            all_metrics.append(metrics)

    results_df = pd.DataFrame(all_metrics)
    results_df.to_csv(RESULTS / "model_comparison.csv", index=False)

    print("\nFINAL MODEL COMPARISON")
    print(
        results_df[
            ["split", "model", "accuracy", "macro_f1",
             "weighted_f1", "train_seconds"]
        ].sort_values(["split", "macro_f1"], ascending=[True, False])
        .to_string(index=False)
    )

    print(f"\nMetrics saved to: {RESULTS / 'model_comparison.csv'}")
    print(f"Models saved to: {MODELS.resolve()}")


if __name__ == "__main__":
    main()
