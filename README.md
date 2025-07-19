# Credit Card Fraud Detection 🕵️‍♂️💳

This project aims to detect fraudulent credit card transactions using basic Exploratory Data Analysis (EDA) and a Random Forest classifier. It is designed as a beginner-friendly project to practice working with imbalanced datasets and applying machine learning for classification.

---

## 🔍 Project Overview

- **Goal:** Classify credit card transactions as fraudulent or genuine.
- **Dataset:** [Credit Card Fraud Detection Dataset (Kaggle)](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Tech Stack:** Python, Pandas, Matplotlib, Seaborn, Scikit-learn

---

## 🧰 Tools & Libraries Used

- `pandas` – for data handling and preprocessing  
- `matplotlib` & `seaborn` – for visualization  
- `sklearn` – for machine learning model (Random Forest), evaluation metrics

---

## 🔢 Steps Followed

1. **Data Loading and Preprocessing**
   - Checked for missing values and data imbalance.
   - Scaled numerical features where needed.
   
2. **Exploratory Data Analysis**
   - Visualized fraud vs non-fraud transactions.
   - Examined transaction amounts and time distributions.
   
3. **Model Training**
   - Used `RandomForestClassifier` to build the model.
   - Trained on the imbalanced dataset without SMOTE.

4. **Model Evaluation**
   - Evaluated using Confusion Matrix and Classification Report.
   - Adjusted prediction threshold to improve recall.

---

## ✅ Results

- Achieved good accuracy and recall using Random Forest.
- Custom thresholding helped in detecting more frauds.
- No hyperparameter tuning or advanced sampling techniques used, keeping the project beginner-friendly.

---



