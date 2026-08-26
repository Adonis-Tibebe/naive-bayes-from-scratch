# Naive Bayes From Scratch

This project implements and evaluates a **Multinomial Naive Bayes sentiment classifier** for the Kaggle Twitter Entity Sentiment Analysis dataset. The classifier, smoothing logic, prediction logic, and evaluation metrics are implemented manually in `src/`; scikit-learn is used only for text vectorization.

The work is organized as a clear experimentation path:

1. Clean and understand the data.
2. Train a baseline count-based Naive Bayes model.
3. Improve the count model with bigrams and stronger document-frequency filtering.
4. Test a TF-IDF + bigram version and compare all results.

## Project Structure

```text
.
|-- data/
|   |-- raw/
|   |   |-- twitter_training.csv
|   |   `-- twitter_validation.csv
|   `-- processed/
|       |-- train_cleaned.csv
|       `-- validation_cleaned.csv
|-- models/
|   |-- naive_bayes_model.pkl
|   |-- vectorizer.pkl
|   |-- confusion_matrix.png
|   |-- Experimentation (Bigrams + Hard IDF)/
|   |   |-- exp_naive_bayes_model.pkl
|   |   |-- exp_vectorizer.pkl
|   |   `-- exp_confusion_matrix.png
|   `-- Tf-Idf version/
|       |-- tfidf_naive_bayes_model.pkl
|       |-- tfidf_vectorizer.pkl
|       `-- tfidf_confusion_matrix.png
|-- notebooks/
|   |-- 01_EDA-Processing.ipynb
|   |-- 01_Model_Training.ipynb
|   |-- 02_test_and_evaluation.ipynb
|   |-- Experimentation (Bigrams + Hard IDF)/
|   |   |-- 01_exp_model_training.ipynb
|   |   `-- 02_exp_test_and_evaluation.ipynb
|   `-- Tf-Idf version/
|       |-- 01_tfidf_model_training.ipynb
|       `-- 02_tfidf_test_and_evaluation.ipynb
|-- src/
|   |-- metrics.py
|   |-- naive_bayes.py
|   |-- preprocessing.py
|   `-- utils.py
|-- requirements.txt
`-- README.md
```

### Directory Significance

- `data/raw/` contains the original training and validation CSV files.
- `data/processed/` contains cleaned files used by all training and evaluation notebooks.
- `src/` contains reusable Python modules. This is where the main implementation lives.
- `notebooks/` documents the workflow, experiments, outputs, and analysis.
- `models/` contains saved vectorizers, trained model objects, and confusion matrix images for each model variation.

## What Is Imported vs Implemented

The project uses standard libraries for file handling, arrays, tables, plots, and vectorization:

- `pandas` is used to load, clean, inspect, and save CSV data.
- `numpy` is used for the Naive Bayes math, log probabilities, matrix operations, and metric calculations.
- `pickle` is used to save and reload trained model/vectorizer artifacts.
- `re` is used for URL and mention normalization.
- `matplotlib` and `seaborn` are used to visualize confusion matrices.
- `CountVectorizer` and `TfidfVectorizer` from scikit-learn are used to convert text into sparse numeric feature matrices.

The following parts are explicitly implemented from scratch:

- Laplace/Lidstone-smoothed log-likelihood calculation.
- Multinomial Naive Bayes class prior estimation.
- Per-class feature likelihood estimation.
- Log-space prediction using posterior scores.
- Model save/load wrapper methods.
- Confusion matrix calculation.
- Accuracy, precision, recall, F1-score, macro averages, and support calculation.
- Cleaning helper for the raw Twitter CSV format.
- Safe validation transformation that uses `.transform()` only, avoiding data leakage.

## Source Scripts

### `src/utils.py`

Defines `load_and_clean_data(file_path)`, a reusable cleaning function for the raw Twitter dataset. It:

- reads raw CSV files without headers;
- assigns the columns `tweet_id`, `entity`, `sentiment`, and `tweet_content`;
- removes the `Irrelevant` class;
- drops missing tweet text;
- removes duplicate tweet content;
- replaces URLs with `<url>`;
- replaces Twitter mentions with `<mention>`;
- returns only `tweet_content` and `sentiment`.

This script is used especially for preparing the validation set after the cleaning logic is explored in the EDA notebook.

### `src/preprocessing.py`

Contains vectorization helpers:

- `fit_transform_train(...)` fits a `CountVectorizer` on training text only and returns both the vectorizer and sparse count matrix.
- `transform_new_data(...)` transforms validation/test data with an already fitted vectorizer. It intentionally does not call `.fit()` or `.fit_transform()`, which prevents validation leakage.
- `fit_transform_tfidf_train(...)` fits a `TfidfVectorizer` for the TF-IDF variation.

The vectorizers keep stopwords because words like `not` and `never` are important sentiment modifiers.

### `src/naive_bayes.py`

Contains the custom model implementation:

- `calculate_smoothed_log_likelihood(...)` applies smoothing:

```text
log P(w_i | c) = log((count(w_i, c) + alpha) / (total_words_c + alpha * vocabulary_size))
```

- `MultinomialNaiveBayes.fit(X, y)` computes class log-priors and smoothed feature log-likelihoods.
- `MultinomialNaiveBayes.predict(X)` computes log-posterior scores with matrix multiplication and returns the highest scoring class.
- `get_learned_parameters()` exposes learned priors and likelihoods for notebook inspection.
- `save_model()` and `load_model()` serialize and reload the trained model.

### `src/metrics.py`

Implements evaluation without relying on scikit-learn metrics:

- `compute_confusion_matrix(...)` builds an actual-vs-predicted class matrix.
- `compute_accuracy(...)` computes overall accuracy.
- `compute_classification_metrics(...)` computes per-class precision, recall, F1-score, support, macro precision, macro recall, macro F1, and accuracy.
- `plot_confusion_matrix(...)` saves a heatmap image for review.

## Notebook Guide

Read the notebooks in this order.

### 1. EDA and preprocessing

Notebook: `notebooks/01_EDA-Processing.ipynb`

This notebook explores and cleans the dataset before modeling. Important observations:

- Training data after cleaning: **57,297 rows**.
- Validation data after cleaning: **827 rows**.
- Training class distribution:
  - Negative: 21,171
  - Positive: 19,078
  - Neutral: 17,048
- Validation class distribution:
  - Neutral: 285
  - Positive: 276
  - Negative: 266
- Raw vocabulary size before vectorizer filtering: **60,264 unique tokens**.
- Twitter-specific noise found:
  - URLs in 2,120 tweets.
  - Mentions in 9,856 tweets.
  - Negation patterns in 9,057 tweets.

The notebook saves `data/processed/train_cleaned.csv` and `data/processed/validation_cleaned.csv`.

### 2. Baseline CountVectorizer model

Training notebook: `notebooks/01_Model_Training.ipynb`  
Evaluation notebook: `notebooks/02_test_and_evaluation.ipynb`

This is the first model variation. It uses:

- `CountVectorizer`
- unigrams only: `ngram_range=(1, 1)`
- `min_df=5`
- `max_df=0.85`
- smoothing: `alpha=1.0`

Training output:

- Matrix shape: **57,297 x 14,250**
- Non-zero entries: **940,116**
- Sparsity: **99.88%**
- Learned classes: `Negative`, `Neutral`, `Positive`

Validation results:

| Metric | Score |
| --- | ---: |
| Accuracy | 0.8174 |
| Macro F1 | 0.8161 |
| Negative F1 | 0.8250 |
| Neutral F1 | 0.7866 |
| Positive F1 | 0.8367 |

This baseline performed reasonably well, but the Neutral class had the weakest recall at **0.6982**, meaning many neutral tweets were pulled into positive or negative predictions.

### 3. CountVectorizer experiment: bigrams + hard IDF-style filtering

Training notebook: `notebooks/Experimentation (Bigrams + Hard IDF)/01_exp_model_training.ipynb`  
Evaluation notebook: `notebooks/Experimentation (Bigrams + Hard IDF)/02_exp_test_and_evaluation.ipynb`

This second variation tests whether phrase-level features and aggressive filtering improve the baseline. It uses:

- `CountVectorizer`
- unigrams and bigrams: `ngram_range=(1, 2)`
- `min_df=10`
- `max_df=0.10`
- smoothing through the same custom Naive Bayes implementation.

Training output:

- Matrix shape: **57,297 x 20,068**
- Non-zero entries: **1,132,156**
- Sparsity: **99.90%**

Validation results:

| Metric | Score |
| --- | ---: |
| Accuracy | 0.8404 |
| Macro F1 | 0.8401 |
| Negative F1 | 0.8386 |
| Neutral F1 | 0.8281 |
| Positive F1 | 0.8537 |

Main analysis:

- Neutral recall improved from **0.6982** to **0.7860**.
- Neutral errors dropped from **86** to **61**.
- Bigrams helped capture phrases such as negations and short sentiment expressions.
- The `max_df` sweep showed that a stricter threshold around `0.10` removed high-frequency domain noise while keeping useful sentiment terms.

### 4. TF-IDF + bigram model

Training notebook: `notebooks/Tf-Idf version/01_tfidf_model_training.ipynb`  
Evaluation notebook: `notebooks/Tf-Idf version/02_tfidf_test_and_evaluation.ipynb`

This third variation tests whether TF-IDF weighting improves over hard count filtering. It uses:

- `TfidfVectorizer`
- unigrams and bigrams: `ngram_range=(1, 2)`
- `min_df=10`
- `max_df=0.50`
- Lidstone smoothing with `alpha=0.05`

Training output:

- Matrix shape: **57,297 x 20,086**
- Sparsity: **99.88%**

Validation results:

| Metric | Score |
| --- | ---: |
| Accuracy | 0.8791 |
| Macro F1 | 0.8791 |
| Negative F1 | 0.8824 |
| Neutral F1 | 0.8734 |
| Positive F1 | 0.8814 |

Main analysis:

- This was the best-performing model.
- TF-IDF improved all three classes compared with the best count-based experiment.
- Neutral precision reached **0.9154**, showing that TF-IDF handled neutral wording better than hard deletion of frequent terms.
- The model kept useful frequent terms but reduced their influence through weighting instead of removing them entirely.

## Model Comparison

| Model Variation | Features | Accuracy | Macro F1 | Key Takeaway |
| --- | --- | ---: | ---: | --- |
| Baseline Count NB | Unigrams, `min_df=5`, `max_df=0.85`, `alpha=1.0` | 0.8174 | 0.8161 | Good starting point, but Neutral recall was weak. |
| Bigram + Hard Filter Count NB | Unigrams+bigrams, `min_df=10`, `max_df=0.10` | 0.8404 | 0.8401 | Improved class balance and reduced Neutral misclassification. |
| TF-IDF Bigram NB | Unigrams+bigrams, `min_df=10`, `max_df=0.50`, `alpha=0.05` | 0.8791 | 0.8791 | Best overall model; strongest and most balanced performance. |

## How To Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Recommended notebook order:

1. `notebooks/01_EDA-Processing.ipynb`
2. `notebooks/01_Model_Training.ipynb`
3. `notebooks/02_test_and_evaluation.ipynb`
4. `notebooks/Experimentation (Bigrams + Hard IDF)/01_exp_model_training.ipynb`
5. `notebooks/Experimentation (Bigrams + Hard IDF)/02_exp_test_and_evaluation.ipynb`
6. `notebooks/Tf-Idf version/01_tfidf_model_training.ipynb`
7. `notebooks/Tf-Idf version/02_tfidf_test_and_evaluation.ipynb`

## Final Summary

The project demonstrates a complete Naive Bayes sentiment classification workflow: data cleaning, feature extraction, custom model implementation, custom evaluation, experimentation, model persistence, and result analysis. The strongest result comes from the **TF-IDF + bigram Naive Bayes model**, which achieved **87.91% accuracy** and **87.91% macro F1** on the cleaned validation set.
