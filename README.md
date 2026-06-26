# 💳 Credit Card Fraud Detection

A machine learning system for real-time credit card fraud detection with 
model explainability. Trained on 284,807 transactions with extreme class 
imbalance (0.17% fraud rate).

![Python](https://img.shields.io/badge/Python-3.10-blue)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![MLflow](https://img.shields.io/badge/Tracking-MLflow-blue)
![Streamlit](https://img.shields.io/badge/App-Streamlit-red)

---

## 🎯 Problem Statement

Credit card fraud detection is a classic extreme imbalance problem — only 
0.17% of transactions are fraudulent. This makes standard accuracy a 
completely useless metric (a model predicting every transaction as 
legitimate gets 99.83% accuracy while catching zero fraud cases).

This project tackles the problem with:
- Proper imbalance handling via SMOTE
- Evaluation using F1, PR-AUC instead of accuracy
- Per-prediction explainability using SHAP
- Interactive Streamlit dashboard for real-time analysis

---

## 📊 Dataset

- **Source:** [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size:** 284,807 transactions
- **Fraud rate:** 0.17% (492 fraud cases)
- **Features:** V1-V28 (PCA transformed), Amount, Time
- **Missing values:** None

---

## 🔍 EDA Key Findings

- Severe class imbalance — accuracy is meaningless as a metric
- Fraud transactions have higher average amount than legitimate ones
- V14, V17, V12 show strongest correlation with fraud
- Amount and Time required StandardScaler (V features already PCA scaled)

---

## ⚙️ Project Pipeline
Raw Data
│
▼

EDA & Analysis          # notebooks/01_eda.ipynb
│
▼

Preprocessing           # notebooks/02_preprocessing.ipynb
├── Scale Amount + Time
├── Train/Test split (80/20 stratified)
└── SMOTE on train set only
│
▼

Model Training          # notebooks/03_modeling.ipynb
├── Logistic Regression (baseline)
├── Random Forest
├── XGBoost ← best model
└── LightGBM
│
▼

Experiment Tracking     # MLflow (localhost:5000)
│
▼

SHAP Explainability     # notebooks/04_shap.ipynb
├── Global feature importance
├── Beeswarm plot
├── Per-transaction waterfall plots
└── Dependence plots
│
▼

Streamlit Dashboard     # app/app.py

---

## 📈 Model Comparison

                       F1  ROC_AUC  PR_AUc
LogisticRegression  0.1066   0.9730  0.7557
Random Forest       0.5663   0.9872  0.8325
XGBoost             0.7688   0.9849  0.8655
LightGBM            0.4315   0.9792  0.7623

## 🏆 Why XGBoost Won

- Best PR-AUC score — most important metric for imbalanced fraud detection
- Handles feature interactions better than linear models
- Gradient boosting ensemble reduces both bias and variance
- SHAP natively supported via TreeExplainer

---

## 🔎 SHAP Explainability

Instead of a black-box prediction, every transaction comes with an 
explanation:

- **Global:** Which features matter most across all predictions
- **Local:** Why this specific transaction was flagged as fraud

Example waterfall plot interpretation:
- V14 = -12.3 pushed fraud probability up significantly
- V17 = -8.1 also pushed toward fraud
- Amount_scaled = 0.2 slightly pushed toward legitimate

This is critical for regulatory compliance — EU AI Act and GDPR Article 22 
require explainability for automated financial decisions.

---

## 🚀 How to Run

**1. Clone the repo**
```bash
git clone https://github.com/Malyank7/credit-fraud-detector.git
cd credit-fraud-detector
```

**2. Install dependencies**
```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

**3. Download dataset**

Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) 
and place in `data/`

**4. Run notebooks in order**
notebooks/01_eda.ipynb

notebooks/02_preprocessing.ipynb

notebooks/03_modeling.ipynb

notebooks/04_shap.ipynb

**5. Launch MLflow dashboard**
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Visit `http://localhost:5000`

**6. Launch Streamlit app**
```bash
cd Credit-fraud-detector
streamlit run ../app/app.py
```
Visit `http://localhost:8501`

---

## 📁 Project Structure
credit-fraud-detector/

├── Credit-fraud-detector

├── data/

│   ├── creditcard.csv          # raw dataset (not tracked by git)

│   └── processed/              # preprocessed numpy arrays

├── notebooks/

│   ├── 01_eda.ipynb

│   ├── 02_preprocessing.ipynb

│   ├── 03_modeling.ipynb

│   └── 04_shap.ipynb

├── src/

│   ├── preprocess.py

│   ├── train.py

│   └── evaluate.py

├── models/                     # saved model files

├── app/

│   └── app.py                  # Streamlit dashboard

├── requirements.txt

└── README.md

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Pandas, NumPy | Data manipulation |
| Scikit-learn | Preprocessing, baseline models |
| XGBoost, LightGBM | Gradient boosting models |
| imbalanced-learn | SMOTE oversampling |
| SHAP | Model explainability |
| MLflow | Experiment tracking |
| Streamlit | Interactive dashboard |

---

## 💡 Key Learnings

1. **Accuracy is misleading** on imbalanced datasets — always use F1 + PR-AUC
2. **SMOTE must only be applied to training data** — never test data
3. **Stratified splits** preserve class ratio across train/test
4. **SHAP explainability** is not optional in regulated domains
5. **PR-AUC > ROC-AUC** for imbalanced problems — true negatives 
   inflate ROC-AUC artificially