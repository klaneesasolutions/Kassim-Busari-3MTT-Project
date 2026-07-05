# Salary Prediction Using the Adult Census Dataset

**Author:** Kassim Busari  
**Assignment:** Machine Learning Model for Salary Prediction  
**Date:** June 2026

---

## 1. Project Overview

This project builds a binary classification model to predict whether an individual's annual income exceeds **$50,000** based on demographic and employment-related features from the **Adult Census Income** dataset (also known as the "Census Income" dataset).

The dataset contains 14 features (both categorical and continuous) and one target variable indicating income level. The notebook demonstrates the full machine learning pipeline, including data preprocessing, model training, and performance evaluation using multiple metrics.

---

## 2. Dataset Description

The dataset was sourced from the UCI Machine Learning Repository. The target variable is **Salary** (or `income`), with two classes:
- `<=50K`
- `>50K`

### Features Used

| Feature | Type | Description |
| :--- | :--- | :--- |
| `age` | Continuous | Age of the individual |
| `workclass` | Categorical | Employment type (Private, Self-emp, Government, etc.) |
| `fnlwgt` | Continuous | Final weight (sampling weight) |
| `education` | Categorical | Highest level of education |
| `education_num` | Continuous | Numeric representation of education level |
| `marital_status` | Categorical | Marital status |
| `occupation` | Categorical | Job type |
| `relationship` | Categorical | Family relationship status |
| `race` | Categorical | Race of the individual |
| `gender` | Categorical | Male or Female |
| `capital_gain` | Continuous | Capital gains recorded |
| `capital_loss` | Continuous | Capital losses recorded |
| `hours_per_week` | Continuous | Number of hours worked per week |
| `native_country` | Categorical | Country of origin |

**Target:** `Salary` (Binary: `<=50K` / `>50K`)

---

## 3. Methodology

### 3.1 Data Preprocessing
1. **Handling Missing Values:**  
   Rows containing missing values (denoted by `?`) were dropped to maintain data integrity.
2. **Encoding Categorical Variables:**  
   One-hot encoding was applied to all categorical features to transform them into a numerical format suitable for machine learning algorithms.
3. **Scaling Numerical Features:**  
   Continuous features (`age`, `fnlwgt`, `education_num`, `capital_gain`, `capital_loss`, `hours_per_week`) were standardized using `StandardScaler` to have zero mean and unit variance.
4. **Target Encoding:**  
   The target column was mapped to binary values: `0` for `<=50K` and `1` for `>50K`.

### 3.2 Model Selection
I chose a **Random Forest Classifier** due to its robustness, ability to handle non-linear relationships, and built-in feature importance. The model was configured with:
- `n_estimators = 100`
- `random_state = 42` (for reproducibility)

### 3.3 Train-Test Split
The preprocessed data was split into:
- **Training Set:** 80% of the data
- **Test Set:** 20% of the data

Stratification was applied to preserve the original class distribution in both splits.

---

## 4. Performance Evaluation

The model was evaluated using several standard classification metrics. The primary three metrics required are **Accuracy**, **Precision**, and **Recall** (for the positive class, `>50K`). Additional metrics (F1-score and ROC-AUC) are provided for a more comprehensive assessment.

### 4.1 Confusion Matrix

| | Predicted `<=50K` | Predicted `>50K` |
| :--- | :---: | :---: |
| **Actual `<=50K`** | 4446 (TN) | 471 (FP) |
| **Actual `>50K`** | 663 (FN) | 453 (TP) |

### 4.2 Calculated Metrics (Test Set)

| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **Accuracy** | **0.8120** (81.20%) | Overall correct predictions. |
| **Precision (Class 1)** | **0.4903** (49.03%) | When the model predicts `>50K`, it is correct ~49% of the time. |
| **Recall (Class 1)** | **0.4059** (40.59%) | The model captures ~41% of the actual `>50K` cases. |
| **F1-score** | 0.4441 | Harmonic mean of precision and recall. |
| **ROC-AUC** | 0.8425 | Good discriminatory ability (chance = 0.5, perfect = 1.0). |

### 4.3 Detailed Classification Report
