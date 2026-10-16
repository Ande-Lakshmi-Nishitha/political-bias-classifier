# Political Bias Classifier

A machine learning project that classifies news articles into **Left, Center, or Right** political bias categories using article text. The project compares traditional text-classification models and evaluates how well they generalize from articles in known media outlets to articles from previously unseen outlets.

## Overview

Political bias classification is a text-classification task in which a model attempts to identify the political leaning associated with a news article.

This project explores two questions:

1. How well can traditional machine learning models classify political bias when articles are randomly split into training and testing sets?
2. How does performance change when the test set contains articles from media outlets that were not present during training?

The project uses TF-IDF text representations with Multinomial Naive Bayes and Linear Support Vector Classification (Linear SVC). It also investigates whether character-level TF-IDF features improve generalization to unseen media outlets.

## Objectives

- Build a baseline political bias classifier using article text.
- Compare Multinomial Naive Bayes and Linear SVC.
- Evaluate models using random and media-disjoint data splits.
- Investigate the contribution of word-level and character-level TF-IDF features.
- Analyze class-level performance using precision, recall, F1-score, and confusion matrices.
- Identify limitations in generalizing across publishers.

## Dataset

The project uses `bias_clean.csv`, containing **10,832 news articles** from approximately **297 media outlets**.

The dataset includes the following fields:

| Column | Description |
|---|---|
| `page_text` | Main article text |
| `title` | Article headline |
| `site` | Publisher or media outlet |
| `bias` | Original political bias label |
| `topic` | Article topic |
| `date` | Publication date |
| `url` | Article URL |

### Label mapping

The original labels are consolidated into three classes:

| Original labels | Final class |
|---|---|
| `left`, `leaning-left` | Left |
| `center` | Center |
| `leaning-right`, `right` | Right |

The resulting class distribution is:

| Class | Articles |
|---|---:|
| Left | 4,881 |
| Center | 1,478 |
| Right | 4,473 |
| **Total** | **10,832** |

The dataset is not balanced across classes, with fewer Center articles than Left or Right articles. This is important when interpreting accuracy and class-specific performance.

**Dataset note:** The raw dataset is not included in the repository by default. Obtain it from the authorized source and place `bias_clean.csv` in the project root.

## Methodology

### 1. Text preprocessing

The pipeline prepares article text for feature extraction. It includes text cleaning, label normalization, and a heuristic to remove exact mentions of publisher names from article text.

Exact-text deduplication is also checked. In the current dataset run, it removed zero rows.

Publisher-name removal is only a heuristic: it does not guarantee that every publisher-related clue is removed.

### 2. Feature extraction

The baseline uses TF-IDF (Term Frequency–Inverse Document Frequency) to convert article text into numerical features.

The word-level representation uses unigram and bigram features, English stop-word removal, a minimum document frequency of 2, and a maximum of 150,000 features.

A second experiment combines word-level TF-IDF with character-level TF-IDF using character n-grams of lengths 3–5.

### 3. Models

**Multinomial Naive Bayes**

A traditional probabilistic text-classification baseline.

**Linear SVC**

A linear support vector classifier trained on TF-IDF features. The baseline uses class-balanced weights to account for differences in class frequency.

**Word + character TF-IDF with Linear SVC**

An additional experiment that combines word and character features to investigate whether character patterns improve classification across unseen publishers.

### 4. Evaluation strategies

**Random split**

Articles are randomly divided into training and testing sets, with stratification by class. Articles from the same publisher may appear in both sets.

**Media-disjoint split**

Publishers are separated between training and testing. The model is evaluated on articles from outlets it did not see during training.

The media-disjoint experiment is a more demanding test of cross-publisher generalization. It helps reveal whether a model learns patterns that transfer across outlets or relies on publisher-specific writing styles and vocabulary.

## Results

### Random split and media-disjoint baseline comparison

The following results are from the recorded model comparison:

| Split | Model | Accuracy | Macro F1 |
|---|---|---:|---:|
| Random | Multinomial Naive Bayes | 0.5699 | 0.3845 |
| Random | Linear SVC | 0.7997 | 0.7660 |
| Media-disjoint | Multinomial Naive Bayes | 0.5121 | 0.3122 |
| Media-disjoint | Linear SVC | 0.5013 | 0.3915 |

### Word + character TF-IDF experiment

The additional word-plus-character Linear SVC model produced the following results on the saved media-disjoint test split:

| Metric | Score |
|---|---:|
| Accuracy | 0.4726 |
| Macro precision | 0.3703 |
| Macro recall | 0.3896 |
| Macro F1 | 0.3630 |
| Weighted F1 | 0.4333 |

The validation experiment showed that adding character features improved validation macro F1 from approximately 0.4362 to 0.4829. However, the word-plus-character model did not outperform the word-only SVC on the recorded final media-disjoint test results.

### Interpretation

- Linear SVC performed substantially better than Naive Bayes on the random split.
- Performance dropped sharply when testing on unseen media outlets.
- The word-only Linear SVC achieved the highest macro F1 among the recorded final media-disjoint comparisons.
- The Center class remained difficult to identify. The word-plus-character model achieved only 0.06 recall for Center on the final media-disjoint test set.
- These results suggest that article-level political bias classification is sensitive to publisher distribution and class imbalance.

The results do not establish that the model has learned political bias independently of publisher-specific language. Further experiments are needed to distinguish these effects.

## Evaluation metrics

The project uses:

- **Accuracy:** Fraction of correctly classified articles.
- **Precision:** Fraction of predictions for a class that are correct.
- **Recall:** Fraction of actual articles in a class that the model identifies.
- **Macro F1:** Average F1-score across classes, weighting each class equally.
- **Weighted F1:** Average F1-score weighted by each class's support.
- **Confusion matrix:** Shows which classes are confused with one another.

Macro F1 is especially useful here because the dataset contains fewer Center articles than Left or Right articles.

## Project structure

The main scripts and generated artifacts are organized as follows. Adjust filenames if your local repository differs.

```text
political-bias-classifier/
├── bias_clean.csv                 # Local dataset; not committed
├── train_evaluate.py              # Baseline models and split evaluation
├── validate_svm.py                # SVM validation experiment
├── validate_char_tfidf.py         # Word vs. word + character features
├── evaluate_final_media.py        # Final media-disjoint evaluation
├── requirements.txt
├── README.md
├── models/
│   └── ...                        # Saved model artifacts; optional
└── results/
    ├── splits/
    │   ├── random_train.csv
    │   ├── random_test.csv
    │   ├── media_train.csv
    │   └── media_test.csv
    ├── model_comparison.csv
    ├── ...                        # Reports, predictions and metrics
```

## Installation

### Requirements

- Python 3.10 or another Python version compatible with the project's dependencies
- pip
- Git

### 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd political-bias-classifier
```

Replace `<YOUR_REPOSITORY_URL>` with the URL of your GitHub repository.

### 2. Create a virtual environment

On Windows:

```cmd
py -3.10 -m venv .venv
.venv\Scripts\activate
```

If Python 3.10 is unavailable, use an installed Python version supported by your dependencies.

### 3. Install dependencies

```cmd
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Add the dataset

Place `bias_clean.csv` in the project root, next to `train_evaluate.py`.

## Running the experiments

Run the baseline comparison:

```cmd
python train_evaluate.py
```

Run the SVM validation experiment:

```cmd
python validate_svm.py
```

Compare word-level and character-level features:

```cmd
python validate_char_tfidf.py
```

Evaluate the word-plus-character model on the media-disjoint test split:

```cmd
python evaluate_final_media.py
```

The scripts should be run in the intended order if later experiments depend on splits or artifacts generated by earlier scripts. Check each script's output paths before running it.


## References

- Baly et al., *Predicting Factuality of Reporting and Bias of News Media Sources*. Use the relevant paper and associated repository documentation for the exact reference and methodology details.
- Project repository: [Article Bias Prediction](https://github.com/ramybaly/Article-Bias-Prediction).
