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
