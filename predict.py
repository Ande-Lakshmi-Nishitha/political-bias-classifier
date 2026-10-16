
from pathlib import Path
import re
import joblib

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"


def clean_article(text, site=None):
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if site and site.strip():
        outlet = re.escape(site.strip().lower())
        text = re.sub(rf"(?<!\w){outlet}(?!\w)", " ", text)

    return text.strip()


def main():
    print("Available models:")
    print("1. Linear SVM - media-based split")
    print("2. Linear SVM - random split")
    choice = input("Choose model (1/2): ").strip()

    model_files = {
        "1": MODELS / "media_linearsvc.joblib",
        "2": MODELS / "random_linearsvc.joblib",
    }

    model_path = model_files.get(choice)
    if model_path is None:
        print("Invalid choice.")
        return

    if not model_path.exists():
        print("Model not found. Run python train_evaluate.py first.")
        return

    article = input("Paste article text: ").strip()
    site = input("Publisher name (optional): ").strip()

    if len(article) < 50:
        print("Please provide a longer article excerpt.")
        return

    model = joblib.load(model_path)
    cleaned = clean_article(article, site or None)

    prediction = model.predict([cleaned])[0]

    print("\nPredicted political leaning:", prediction)

    if hasattr(model, "decision_function"):
        scores = model.decision_function([cleaned])[0]
        print("Raw SVM decision scores:", scores)
        print("These are not calibrated probabilities.")


if __name__ == "__main__":
    main()
