# News Article Classification with N-Gram Models

## Overview

This project builds and compares multiple text classification models to categorize news articles from the AG News dataset into four categories: **World**, **Sports**, **Business**, and **Sci/Tech**. The primary goal is to understand how feature extraction choices (unigrams vs. bigrams, raw counts vs. TF-IDF) affect model performance and interpretability.

## Models Compared

We train and evaluate four models using XGBoost with different vectorization schemes:
- Counts of Unigrams
- Counts of Unigrams + Bigrams
- TF-IDF of Unigrams
- TF-IDF of Unigrams + Bigrams

## Key Results

- **Unigram counts** performed best overall based on macro F1 score.
- Identified and visualized SHAP feature importance of top terms for classifying each of the four categories.

## Directory Structure

| File | Description |
|------|-------------|
| `news_classification.ipynb` | Main notebook with model training, evaluation, and visualization. |
| `preprocess.py` | Text preprocessing functions using spaCy. |
| `skopt_hyperparam_tune.py` | Code for Bayesian hyperparameter tuning of XGBoost models. |
| `nltk_code_preprocess.py` | Alternative preprocessing functions using NLTK; not used in main notebook. |
| `README.md` | Project documentation and usage guide. |



## Dependencies

- Python 3.9+
- **spacy**
- **scikit-learn, xgboost, shap, skopt**
- pandas, numpy
- matplotlib, seaborn
- (Optional) nltk
