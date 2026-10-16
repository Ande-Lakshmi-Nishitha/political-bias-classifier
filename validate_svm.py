
from pathlib import Path
import json

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

LABELS = ["left", "center", "right"]
SEED = 42


def main():
    # Reconstruct the same article-cleaning procedure as the main script.
    import train_evaluate as base

    df = base.load_data()

    # Start with the training articles from the existing media split.
    train_split = pd.read_csv(RESULTS / "splits/media_train.csv")
    df_train = df.iloc[train_split["row_index"]].reset_index(drop=True)

    # Split by outlet, never by individual article.
    splitter = GroupShuffleSplit(
        n_splits=100,
        test_size=0.20,
        random_state=SEED,
    )

    chosen = None
    for tr_idx, val_idx in splitter.split(
        df_train,
        df_train["label"],
        groups=df_train["site"],
    ):
        if (
            set(df_train.iloc[tr_idx]["label"]) == set(LABELS)
            and set(df_train.iloc[val_idx]["label"]) == set(LABELS)
        ):
            chosen = (tr_idx, val_idx)
            break

    if chosen is None:
        raise RuntimeError(
            "Could not find a validation split with all classes. "
            "Inspect label coverage by outlet."
        )

    tr_idx, val_idx = chosen
    train = df_train.iloc[tr_idx]
    val = df_train.iloc[val_idx]

    overlap = set(train["site"]) & set(val["site"])
    assert not overlap, f"Outlet leakage: {overlap}"

    print("Training articles:", len(train))
    print("Validation articles:", len(val))
    print("Training outlets:", train["site"].nunique())
    print("Validation outlets:", val["site"].nunique())
    print("Outlet overlap:", len(overlap))

    print("\nTraining label proportions:")
    print(train["label"].value_counts(normalize=True).round(3))
    print("\nValidation label proportions:")
    print(val["label"].value_counts(normalize=True).round(3))

    # The second model gives center more weight than the other classes.
    configurations = {
        "balanced": {
            "left": 1.0,
            "center": 1.0,
            "right": 1.0,
        },
        "center_sensitive": {
            "left": 1.0,
            "center": 2.0,
            "right": 1.0,
        },
    }

    rows = []

    for name, weights in configurations.items():
        model = Pipeline([
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=150_000,
                    sublinear_tf=True,
                    dtype="float32",
                ),
            ),
            (
                "classifier",
                LinearSVC(
                    C=1.0,
                    class_weight=weights,
                    random_state=SEED,
                    max_iter=5000,
                ),
            ),
        ])

        model.fit(train["text"], train["label"])
        pred = model.predict(val["text"])

        report = classification_report(
            val["label"],
            pred,
            labels=LABELS,
            output_dict=True,
            zero_division=0,
        )

        result = {
            "model": name,
            "accuracy": accuracy_score(val["label"], pred),
            "macro_f1": f1_score(
                val["label"], pred,
                labels=LABELS,
                average="macro",
                zero_division=0,
            ),
            "center_recall": report["center"]["recall"],
            "center_f1": report["center"]["f1-score"],
        }
        rows.append(result)

        print(f"\n{name.upper()}")
        print(json.dumps(result, indent=2))
        print(classification_report(
            val["label"], pred,
            labels=LABELS,
            zero_division=0,
        ))

    results = pd.DataFrame(rows)
    results.to_csv(
        RESULTS / "svm_validation_comparison.csv",
        index=False,
    )
    print("\nSaved results/svm_validation_comparison.csv")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
