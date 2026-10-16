
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
MODELS = ROOT / "models"

LABELS = ["left", "center", "right"]


def build_model():
    word_features = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_features=150_000,
        sublinear_tf=True,
        dtype="float32",
    )

    char_features = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=150_000,
        sublinear_tf=True,
        dtype="float32",
    )

    features = FeatureUnion([
        ("word", word_features),
        ("char", char_features),
    ])

    return Pipeline([
        ("tfidf", features),
        ("classifier", LinearSVC(
            C=1.0,
            class_weight="balanced",
            random_state=42,
            max_iter=5000,
        )),
    ])


def main():
    import train_evaluate as base

    df = base.load_data()

    train_path = RESULTS / "splits" / "media_train.csv"
    test_path = RESULTS / "splits" / "media_test.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "Missing saved media split files. "
            "Run train_evaluate.py first."
        )

    train_indices = pd.read_csv(train_path)["row_index"].tolist()
    test_indices = pd.read_csv(test_path)["row_index"].tolist()

    train = df.iloc[train_indices]
    test = df.iloc[test_indices]

    overlap = set(train["site"]) & set(test["site"])
    if overlap:
        raise RuntimeError(
            f"Train/test outlet leakage detected: {overlap}"
        )

    print("Training articles:", len(train))
    print("Test articles:", len(test))
    print("Training outlets:", train["site"].nunique())
    print("Test outlets:", test["site"].nunique())
    print("Outlet overlap:", len(overlap))

    model = build_model()
    model.fit(train["text"], train["label"])
    predictions = model.predict(test["text"])

    metrics = {
        "model": "word_plus_char_LinearSVC",
        "accuracy": accuracy_score(test["label"], predictions),
        "macro_precision": precision_score(
            test["label"], predictions,
            labels=LABELS, average="macro", zero_division=0,
        ),
        "macro_recall": recall_score(
            test["label"], predictions,
            labels=LABELS, average="macro", zero_division=0,
        ),
        "macro_f1": f1_score(
            test["label"], predictions,
            labels=LABELS, average="macro", zero_division=0,
        ),
        "weighted_f1": f1_score(
            test["label"], predictions,
            average="weighted", zero_division=0,
        ),
    }

    print("\nFINAL MEDIA TEST METRICS")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("\nCLASSIFICATION REPORT")
    print(classification_report(
        test["label"],
        predictions,
        labels=LABELS,
        zero_division=0,
    ))

    report = classification_report(
        test["label"],
        predictions,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).T.to_csv(
        RESULTS / "final_media_word_char_report.csv"
    )

    output = test[
        ["url", "title", "site", "bias", "label"]
    ].copy()
    output["predicted_label"] = predictions
    output["correct"] = output["label"] == output["predicted_label"]
    output.to_csv(
        RESULTS / "final_media_word_char_predictions.csv",
        index=False,
    )

    cm = confusion_matrix(
        test["label"], predictions, labels=LABELS
    )
    pd.DataFrame(
        cm, index=LABELS, columns=LABELS
    ).to_csv(RESULTS / "final_media_word_char_confusion_matrix.csv")

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
    plt.title("Final Media Test: Word + Character TF-IDF SVM")
    plt.tight_layout()
    plt.savefig(
        RESULTS / "final_media_word_char_confusion_matrix.png",
        dpi=160,
    )
    plt.close()

    MODELS.mkdir(exist_ok=True)
    joblib.dump(
        model, MODELS / "final_media_word_char_svm.joblib"
    )

    pd.DataFrame([metrics]).to_csv(
        RESULTS / "final_media_word_char_metrics.csv",
        index=False,
    )

    print("\nSaved final model and metrics to models/ and results/.")


if __name__ == "__main__":
    main()
