# Credit Card Fraud Detection: Resampling & Model Evaluation

This repository provides a comprehensive Python pipeline for exploring how various data resampling techniques impact the performance of multiple classification models in the context of credit card fraud detection. The workflow is reusable for any highly imbalanced classification data and facilitates a data-driven approach to handling class imbalance.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [How It Works](#how-it-works)
- [Results Example](#results-example)
- [Author & License](#author--license)

---

## Overview

Imbalanced datasets, such as those in fraud detection, hinder conventional machine learning models due to the rarity of the minority class. This project evaluates different resampling (balancing) methods and classification models, ultimately generating a matrix that reveals which techniques yield the best detection accuracy.

**Process Summary:**
- Balances the data using SMOTE and Random Undersampling.
- Generates five sample datasets of various statistically-defined sizes.
- Applies five different balancing (sampling) strategies to each train set.
- Trains and tests five machine learning models per sampler.
- Aggregates results for analysis and comparison.

---

## Features

- **Resampling Methods:** Random Under/Over Sampling, SMOTE, SMOTEENN, SMOTETomek
- **Classifiers:** Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, k-Nearest Neighbors
- **Statistical Sampling:** Calculates sample sizes using finite population correction and multiple confidence intervals
- **Result Aggregation:** Averages test accuracies across independent samples for robust comparison
- **Output:** Accuracy matrix, best sampler per model (printed in console)

---

## Requirements

- Python 3.8+
- Packages:  
  - pandas  
  - numpy  
  - scikit-learn  
  - imbalanced-learn

*See `requirements.txt` for easy installation.*

---

## Quick Start

1. **Clone this repository:**
- git clone https://github.com/abhirajsinghjhajj/Credit-Card-Fraud-Detection

2. **Install dependencies:**
- pip install -r requirements.txt

3. **Add your dataset:**
- Place `Creditcard_data.csv` in a `data/` directory.

4. **Run the pipeline:**
- python src/main.py


---

## How It Works

1. **Load Data & Initial Balancing:**  
- Reads credit card data.
- Balances the set by oversampling (SMOTE) to achieve a 10% minority ratio, then undersamples the majority class for parity.

2. **Sample Size Calculation:**  
- Five sample sizes are computed using statistical formulas for finite populations at various error margins (from 5% to 1%).

3. **Repeat Experiments:**  
- For each sample size subset:
  - Split into train/test (70/30, stratified)
  - Resample training data using each of the five strategies
  - Train five different models
  - Evaluate and record test accuracy

4. **Result Aggregation & Matrix Output:**  
- Average results across all samples.
- Print a matrix showing models (rows) vs. samplers (columns).
- Print the best sampler per model.

---

## Results Example

Balanced dataset size: 152
Adjusted sample sizes: [109, 121, 133, 143, 149]

Accuracy matrix (rows = models, columns = samplers):
    Sampling1  Sampling2  Sampling3  Sampling4  Sampling5
M1      0.878      0.883      0.878      0.768      0.884
M2      0.887      0.887      0.877      0.708      0.888
M3      0.967      0.951      0.951      0.829      0.956
M4      0.905      0.905      0.905      0.720      0.908
M5      0.674      0.668      0.668      0.606      0.663

Best sampler per model:
M1 Sampling5
M2 Sampling5
M3 Sampling1
M4 Sampling5
M5 Sampling1


*Here, "Sampling4" refers to SMOTEENN—typically the most effective under these conditions.*

---

## Author & License

**Author:**  
- Your Name (add your details here)

**License:**  
- [MIT License](./LICENSE)

---

**Note:**  
For further analysis, consider extending the script to include other metrics such as Recall, Precision, F1-score, and AUC—crucial for real-world fraud detection beyond simple accuracy.

---
