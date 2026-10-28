# Customer Churn Prediction with ANNs

Two churn-prediction notebooks, one for telecom and one for banking, comparing a neural network with XGBoost.

## Why this exists
Practice with artificial neural networks on tabular data, and a check of whether they beat gradient boosting.

## What it does
**`ann1.ipynb`: telecom churn** (Kaggle `blastchar/telco-customer-churn`, 7,043 customers, 21 columns)
- Cleans values, converts `TotalCharges` to numeric, one-hot encodes categories, min-max scales numbers
- Trains a Keras network (27 → 15 → 10 → 1, ReLU, sigmoid output) for 100 epochs
- Compares it with XGBoost: ANN reached 85.6% training accuracy, XGBoost 77.5% test accuracy

**`ann2.ipynb`: bank churn** (Kaggle `gauravtopre/bank-customer-churn-dataset`, 10,000 customers, 12 columns)
- Drops the ID, scales numeric columns, one-hot encodes gender and country, splits 80/20
- Trains a deeper network (13 → 10 → 9 → 8 → 7 → 1) for 60 epochs
- XGBoost did better: 95.9% training accuracy against 86.5% for the ANN

## Project structure
```
ann/
├── ann1.ipynb  Telecom churn: ANN vs XGBoost
└── ann2.ipynb  Bank churn: ANN vs XGBoost
```

## Getting started
Requirements: Python 3.10+, Jupyter, a Kaggle account for `kagglehub` downloads.
```bash
git clone https://github.com/abhijha8287/ann.git
cd ann
pip install jupyter pandas numpy matplotlib seaborn scikit-learn tensorflow xgboost kagglehub
jupyter notebook ann1.ipynb
```
The notebooks download their datasets through `kagglehub`.

## Tech stack
Python, TensorFlow/Keras, XGBoost, scikit-learn, pandas, Matplotlib, Seaborn

## Notes
Accuracies are as reported in the notebooks. Some are training accuracies, so compare them with care.
