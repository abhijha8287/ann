# ANN Customer Churn Prediction Projects

This repository contains two comprehensive machine learning projects focused on customer churn prediction using Artificial Neural Networks (ANN) and XGBoost classifiers.

## 📁 Project Overview

### ann1.ipynb - Telco Customer Churn Prediction

**Project Description:**
A complete churn prediction analysis using the Telco Customer Churn dataset to predict whether customers will leave the telecom service. The project demonstrates end-to-end machine learning workflow from data preprocessing to model evaluation.

**Dataset:**
- **Source:** Kaggle - "blastchar/telco-customer-churn"
- **Size:** 7,043 customers with 21 features
- **Target Variable:** Churn (binary classification)
- **Key Features:** CustomerID, gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges

**Main Steps:**
1. **Data Loading & Exploration:** Load dataset using kagglehub and perform initial data analysis
2. **Data Cleaning:** Handle missing values and data type conversions (TotalCharges to numeric)
3. **Exploratory Data Analysis (EDA):** Analyze churn patterns by tenure and other features
4. **Data Preprocessing:**
   - Replace inconsistent values ('No internet service' → 'No', 'No phone service' → 'No')
   - Convert categorical variables to binary (Yes/No → 1/0)
   - One-hot encoding for categorical features (InternetService, Contract, PaymentMethod, gender)
   - MinMax scaling for numerical features (tenure, MonthlyCharges, TotalCharges)
5. **Model Building:**
   - **ANN Architecture:** Sequential model with layers (27→15→10→1)
   - **Activation Functions:** ReLU for hidden layers, Sigmoid for output
   - **Training:** 100 epochs with Adam optimizer and binary crossentropy loss
6. **Model Comparison:** Compare ANN performance with XGBoost classifier

**Model Results:**
- **ANN Training Accuracy:** 85.6%
- **XGBoost Test Accuracy:** 77.5%
- **ANN Classification Report:**
  - Precision: 85% (weighted avg)
  - Recall: 86% (weighted avg)
  - F1-Score: 85% (weighted avg)

---

### ann2.ipynb - Bank Customer Churn Prediction

**Project Description:**
A sophisticated churn prediction system for banking customers using the Bank Customer Churn dataset. This project focuses on predicting customer attrition in the banking sector with advanced neural network techniques.

**Dataset:**
- **Source:** Kaggle - "gauravtopre/bank-customer-churn-dataset"
- **File:** Bank Customer Churn Prediction.csv
- **Size:** 10,000 customers with 12 features
- **Target Variable:** Churn (binary classification)
- **Key Features:** customer_id, credit_score, country, gender, age, tenure, balance, products_number, credit_card, active_member, estimated_salary, churn

**Main Steps:**
1. **Data Loading:** Download and load dataset using kagglehub integration
2. **Data Exploration:** 
   - Dataset shape: (10,000, 12)
   - Data types analysis and descriptive statistics
   - Remove customer_id as it's not predictive
3. **Data Preprocessing:**
   - MinMax scaling for numerical features (credit_score, age, tenure, balance, estimated_salary)
   - One-hot encoding for categorical variables (gender, country)
   - Final dataset shape: (10,000, 13) after encoding
4. **Train-Test Split:** 80-20 split with random_state=42
5. **Neural Network Architecture:**
   - **Deep ANN Model:** Sequential architecture (13→10→9→8→7→1)
   - **Activation Functions:** ReLU for hidden layers, Sigmoid for output
   - **Training:** 60 epochs with Adam optimizer
   - **Loss Function:** Binary crossentropy
6. **Model Evaluation & Comparison:** Compare ANN with XGBoost performance

**Model Results:**
- **ANN Training Accuracy:** 86.5%
- **XGBoost Training Accuracy:** 95.9%
- **Training Configuration:** 8,000 training samples, 2,000 test samples
- **Model Performance:** XGBoost significantly outperformed ANN on this dataset

---

## 🛠️ Technologies Used

- **Python Libraries:**
  - `pandas` & `numpy` - Data manipulation and analysis
  - `matplotlib` & `seaborn` - Data visualization
  - `scikit-learn` - Data preprocessing and metrics
  - `tensorflow/keras` - Neural network implementation
  - `xgboost` - Gradient boosting classifier
  - `kagglehub` - Dataset downloading

- **Machine Learning Techniques:**
  - Artificial Neural Networks (ANN)
  - XGBoost Classification
  - Feature scaling (MinMaxScaler)
  - One-hot encoding
  - Train-test splitting
  - Model evaluation metrics

## 📊 Key Insights

1. **Dataset Comparison:**
   - Telco dataset: 7,043 samples, more complex feature engineering required
   - Bank dataset: 10,000 samples, cleaner data with fewer preprocessing steps

2. **Model Performance:**
   - **ann1.ipynb:** ANN (85.6%) > XGBoost (77.5%)
   - **ann2.ipynb:** XGBoost (95.9%) > ANN (86.5%)

3. **Architecture Differences:**
   - Telco model: Simpler architecture (27→15→10→1)
   - Bank model: Deeper architecture (13→10→9→8→7→1)

## 🚀 Getting Started

1. Clone the repository
2. Install required dependencies
3. Run the Jupyter notebooks in Google Colab or local environment
4. Each notebook includes dataset download via kagglehub

## 📈 Future Improvements

- Hyperparameter tuning for both models
- Cross-validation implementation
- Feature importance analysis
- Advanced neural network architectures (LSTM, attention mechanisms)
- Ensemble methods combining ANN and XGBoost

---

**Author:** abhijha8287  
**License:** Open source project for educational purposes
