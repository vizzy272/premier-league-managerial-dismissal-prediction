# premier-league-managerial-dismissal-prediction
MSc Data Science project investigating the prediction of Premier League managerial dismissals using machine learning.
# Premier League Managerial Dismissal Prediction

# Project Overview

This repository contains the original code and supporting materials for an MSc Data Science project investigating the prediction of managerial dismissals in the English Premier League using machine learning.

The study examines whether team performance indicators can be used to identify patterns associated with managerial dismissal.

# Research Aim

To investigate whether machine learning techniques can effectively predict managerial dismissals in the English Premier League using team performance indicators.

# Dataset

The study uses historical English Premier League data covering the 2010/11 to 2024/25 seasons.

The analysis combines publicly available football performance data with managerial dismissal records.

The main predictor variables used in the final models are:

- Wins
- Draws
- Goals Against

The target variable is:

- Dismissed

where 0 represents managers who were not dismissed and 1 represents managers who were dismissed.

# Machine Learning Models

Three supervised classification algorithms were developed and evaluated:

1. Logistic Regression
2. Decision Tree
3. Random Forest

# Model Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

# Main Findings

The Decision Tree and Random Forest models achieved an overall accuracy of 68%, while Logistic Regression achieved an accuracy of 65%.

Logistic Regression achieved the highest ROC-AUC of 0.70.

However, all three models demonstrated limited ability to identify actual managerial dismissal cases, indicating that team performance indicators alone are insufficient for reliably predicting managerial turnover.

# Repository Contents

- `managerial_dismissal_analysis.ipynb` – Main analysis notebook containing data preparation, exploratory analysis, model development and evaluation.
- `requirements.txt` – Python packages required to reproduce the analysis.
- `figures/` – Selected figures generated during the analysis.
- `data/README.md` – Information about the datasets and their sources.

# Reproducibility

The analysis was conducted using Python and the Scikit-learn machine learning library.

The dataset itself is not included in this repository where redistribution may not be permitted. The data sources and preparation procedures are documented separately.

