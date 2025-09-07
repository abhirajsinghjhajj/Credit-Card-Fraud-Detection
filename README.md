# Credit Card Fraud Detection

This project implements a machine learning pipeline for detecting credit card fraud using various resampling techniques and classifiers.

## ✅ Project Overview

The goal is to handle extreme class imbalance in the credit card fraud dataset and evaluate different sampling strategies combined with machine learning models to detect fraudulent transactions effectively.

---

## 🗂 Dataset

The dataset used is the **[Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)** from Kaggle.  
📊 It contains transactions made by credit cards in September 2013 by European cardholders.

- **Rows**: 284,807 transactions  
- **Columns**: 31 (including anonymized PCA features `V1` to `V28`, `Time`, `Amount`, and target `Class`)  
- **Imbalance**: Only 0.172% of transactions are fraudulent.

⚠️ Due to its large size, the dataset is **not included** in this repository.  
👉 Download it from Kaggle:  
[https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

Place the file `creditcard.csv` in the project root before running the code.

---

## ⚙️ How It Works

1. The dataset is loaded and resampled to balance the classes using:
    - SMOTE (Synthetic Minority Over-sampling Technique)  
    - Random Under-Sampler  

2. Multiple dataset sample sizes are calculated based on statistical sample size formula.

3. Various samplers are applied:
    - RandomUnderSampler  
    - RandomOverSampler  
    - SMOTE  
    - SMOTEENN  
    - SMOTETomek  

4. Multiple machine learning models are trained and evaluated:
    - Logistic Regression  
    - Decision Tree  
    - Random Forest  
    - Gradient Boosting  
    - K-Nearest Neighbors  

5. StandardScaler is applied to normalize features.

6. Model performance is evaluated using **accuracy averaged over five different dataset samples**.

---

## 🚀 Getting Started

1. Clone the repository:
    ```bash
    git clone <your-repo-url>
    cd <repo-folder>
    ```

2. Download the dataset from Kaggle:  
    [https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

3. Place `creditcard.csv` in the project root.

4. Install dependencies:
    ```bash
    pip install pandas numpy scikit-learn imbalanced-learn
    ```

5. Run the code:
    ```bash
    python your_script.py
    ```

---

## ✅ Output

- The final result is an accuracy matrix showing performance of all model + sampler combinations.
- It prints the best sampler for each model.

---

## 📚 Notes

- Accuracy may not be the best evaluation metric for highly imbalanced data.  
- Consider using **F1-score, Precision, Recall, or ROC-AUC** for better evaluation.
  
---

## 📜 License

This project is open-source under the MIT License.
