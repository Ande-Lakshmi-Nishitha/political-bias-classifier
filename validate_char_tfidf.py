
from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
MODELS = ROOT / "models"

RESULTS.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)

LABELS = ["left", "center", "right"]
SEED = 42


def make_vectorizer(use_char_features):
    word_vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_features=150_000,
        sublinear_tf=True,
        dtype="float32",
    )

    if not use_char_features:
        return word_vectorizer

    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=150_000,
        sublinear_tf=True,
        dtype="float32",
    )

    return FeatureUnion([
        ("word", word_vectorizer),
        ("char", char_vectorizer),
    ])


def build_model(use_char_features):
    return Pipeline([
        ("tfidf", make_vectorizer(use_char_features)),
        (
            "classifier",
            LinearSVC(
                C=1.0,
                class_weight="balanced",
                random_state=SEED,
                max_iter=5000,
            ),
        ),
    ])


def main():
    # Reuse exactly the same preprocessing as the original experiment.
    import train_evaluate as base

    df = base.load_data()

    # Reuse the existing media-training partition.
    split_path = RESULTS / "splits" / "media_train.csv"
    if not split_path.exists():
        raise FileNotFoundError(
            "Missing results/splits/media_train.csv. "
            "Run train_evaluate.py first."
        )

    split = pd.read_csv(split_path)
    training_pool = df.iloc[split["row_index"]].reset_index(drop=True)

    # Reconstruct the same deterministic outlet-disjoint validation split
    # used in validate_svm.py.
    splitter = GroupShuffleSplit(
        n_splits=100,
        test_size=0.20,
        random_state=SEED,
    )

    chosen = None
    for train_idx, val_idx in splitter.split(
        training_pool,
        training_pool["label"],
        groups=training_pool["site"],
    ):
        if (
            set(training_pool.iloc[train_idx]["label"]) == set(LABELS)
            and set(training_pool.iloc[val_idx]["label"]) == set(LABELS)
        ):
            chosen = (train_idx, val_idx)
            break

    if chosen is None:
        raise RuntimeError(
            "Could not reproduce the outlet-disjoint validation split."
        )

    train_idx, val_idx = chosen
    train = training_pool.iloc[train_idx]
    val = training_pool.iloc[val_idx]

    overlap = set(train["site"]) & set(val["site"])
    if overlap:
        raise RuntimeError(f"Outlet overlap found: {overlap}")

    print("Training articles:", len(train))
    print("Validation articles:", len(val))
    print("Training outlets:", train["site"].nunique())
    print("Validation outlets:", val["site"].nunique())
    print("Outlet overlap:", len(overlap))

    # Both models use the same split, labels and SVM configuration.
    experiments = {
        "word_only": False,
        "word_plus_char": True,
    }

    all_results = []

    for name, use_char in experiments.items():
        print(f"\n{'=' * 60}")
        print("Running:", name)
        print(f"{'=' * 60}")

        model = build_model(use_char)

        model.fit(train["text"], train["label"])
        predictions = model.predict(val["text"])

        report = classification_report(
            val["label"],
            predictions,
            labels=LABELS,
            output_dict=True,
            zero_division=0,
        )

        metrics = {
            "model": name,
            "accuracy": accuracy_score(val["label"], predictions),
            "macro_f1": f1_score(
                val["label"],
                predictions,
                labels=LABELS,
                average="macro",
                zero_division=0,
            ),
            "weighted_f1": f1_score(
                val["label"],
                predictions,
                average="weighted",
                zero_division=0,
            ),
            "center_precision": report["center"]["precision"],
            "center_recall": report["center"]["recall"],
            "center_f1": report["center"]["f1-score"],
        }

        all_results.append(metrics)

        print(json.dumps(metrics, indent=2))
        print("\nPer-class report:")
        print(classification_report(
            val["label"],
            predictions,
            labels=LABELS,
            zero_division=0,
        ))

        # Save reports and predictions for inspection.
        pd.DataFrame(report).T.to_csv(
            RESULTS / f"{name}_validation_report.csv"
        )

        output = val[
            ["url", "title", "site", "bias", "label"]
        ].copy()
        output["predicted_label"] = predictions
        output["correct"] = output["label"] == output["predicted_label"]
        output.to_csv(
            RESULTS / f"{name}_validation_predictions.csv",
            index=False,
        )

        pd.DataFrame(
            confusion_matrix(
                val["label"], predictions, labels=LABELS
            ),
            index=LABELS,
            columns=LABELS,
        ).to_csv(
            RESULTS / f"{name}_validation_confusion_matrix.csv"
        )

        # Save the fitted pipeline.
        joblib.dump(model, MODELS / f"{name}_svm.joblib")

    comparison = pd.DataFrame(all_results)
    comparison.to_csv(
        RESULTS / "word_char_validation_comparison.csv",
        index=False,
    )

    print("\nFINAL COMPARISON")
    print(comparison.to_string(index=False))
    print("\nSaved: results/word_char_validation_comparison.csv")


if __name__ == "__main__":
    main()
