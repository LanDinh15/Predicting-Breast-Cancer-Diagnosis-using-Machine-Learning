## Predicting Breast Cancer Diagnosis Using Machine Learning

### Overview
This project aims to predict whether a breast tumor is malignant (cancerous) or benign (non-cancerous) using machine learning models applied to the Wisconsin Diagnostic Breast Cancer (WDBC) dataset from the UCI Machine Learning Repository. The goal is to assist medical professionals in achieving faster and more reliable diagnoses through a binary classification approach.

### Dataset
- **Source**: UCI Machine Learning Repository
- **Features**: 30 numerical features (e.g., radius, texture, perimeter) grouped into mean, standard error, and worst values.
- **Target**: Diagnosis (M = malignant, B = benign)
- **Rows**: 569 entries
- **File**: `wdbc.data` (raw), `wdbc_cleaned.csv` (preprocessed)

### Features
- **Data Preprocessing**: Drops irrelevant ID column, encodes target variable (M=1, B=0), and applies feature scaling.
- **Exploratory Data Analysis (EDA)**: Visualizes diagnosis distribution and feature correlations.
- **Model Pipeline**: Includes StandardScaler, SelectFromModel (Random Forest-based feature selection), and multiple classifiers.
- **Classifiers Tested**: Logistic Regression, K-Nearest Neighbors (KNN), Decision Tree, Gaussian Naive Bayes, Random Forest, XGBoost, Artificial Neural Network (ANN via MLPClassifier), Support Vector Machine (SVM Linear).
- **Evaluation Metrics**: Accuracy, confusion matrix, precision, recall, F1-score.
- **Out-of-Sample Testing**: Synthetic data with noise to assess model robustness.

### Results
- **Best Model**: SVM Linear (97.2% accuracy, 0.98 recall for malignant cases).
- **Key Findings**:
   - High recall ensures nearly all malignant cases are detected, critical for medical applications.
   - Feature scaling and selection improved performance by handling correlated features (e.g., radius_worst, perimeter_worst).
   - Out-of-sample predictions (with synthetic noise) showed 80% consistency, indicating robustness but some sensitivity to variations.
- **Implications**: The model is a reliable tool for early breast cancer detection, suitable for assisting radiologists, though it requires validation with real patient data.

