# Diabetes Prediction using Machine Learning (Pima Indians Dataset)

This repository contains the implementation and paper for a machine learning project focused on early diabetes prediction using clinical data from the Pima Indians Diabetes dataset. The study includes full steps of data preprocessing, exploratory data analysis (EDA), model training, performance evaluation, and visual interpretation of medical features influencing diabetes outcomes.

---

## 🎯 Project Goal

To build and compare different machine learning classifiers to predict whether a patient is diabetic (1) or non-diabetic (0), supporting early screening and helping reduce serious health risks.

---

## 📊 Dataset Description

- **Name:** Pima Indians Diabetes Database  
- **Total Samples:** 768 patients  
- **Population:** Adult female patients of Pima Indian heritage  
- **Features (8):**
  - Pregnancies
  - Glucose
  - Blood Pressure
  - Skin Thickness
  - Insulin
  - BMI (Body Mass Index)
  - Diabetes Pedigree Function
  - Age
- **Target Label:**
  - `Outcome` → (0 = Non-diabetic, 1 = Diabetic)

> Source: Public dataset available on Kaggle and UCI Machine Learning Repository

This dataset is widely used in the medical data science community due to its relevance and predictive value.

---

## 🧠 Methodology

### 1️⃣ Data Preprocessing
- Handling missing/illogical values (e.g., zero glucose or zero BMI)
- Replacing invalid values with median per feature
- **StandardScaler** applied for normalization
- Stratified **80/20 train-test split**

### 2️⃣ Exploratory Data Analysis
- Summary statistics and visualization of medical features
- Heatmap to examine correlation patterns
- Boxplots to observe distribution differences between diabetic vs non-diabetic groups

### 3️⃣ Machine Learning Models Used
| Model | Description |
|-------|-------------|
| Logistic Regression | Baseline linear classifier |
| Support Vector Machine (RBF) | Handles nonlinear patterns |
| Random Forest | Ensemble model with best performance |

### 4️⃣ Evaluation Metrics
- Accuracy  
- Precision  
- Recall  
- F1-score  
- ROC Curve  
- Confusion Matrix  
- Feature Importance Visualization  

---

## 🚀 Results

| Model | Accuracy | Precision | Recall | F1-score |
|------|:--------:|:---------:|:------:|:--------:|
| Logistic Regression | 70.78% | 60.00% | 50.00% | 54.55% |
| SVM (RBF) | 74.03% | 65.21% | 55.56% | 60.00% |
| **Random Forest** | **77.92%** | **71.73%** | **61.11%** | **66.00%** |

✔ Random Forest achieved the **best predictive performance**  
✔ Glucose & BMI found to be the **most influential predictors**  
✔ Results align with real medical evidence

---

## 📁 Repository Structure

```text
.
├── data/
│   └── diabetes.csv
├── notebooks/
│   └── diabetes_prediction.ipynb
├── figures/
│   ├── heatmap.png
│   ├── boxplots.png
│   ├── confusion_matrix_rf.png
│   ├── roc_curve.png
│   └── feature_importance.png
├── paper/
│   └── Diabetes_Prediction_IEEE_Paper.pdf
└── README.md


# Diabetes Prediction using Machine Learning (Pima Indians Dataset)

This repository contains the implementation and paper for a machine learning project focused on early diabetes prediction using clinical data from the Pima Indians Diabetes dataset. The study includes full steps of data preprocessing, exploratory data analysis (EDA), model training, performance evaluation, and visual interpretation of medical features influencing diabetes outcomes.

---

## 🎯 Project Goal

To build and compare different machine learning classifiers to predict whether a patient is diabetic (1) or non-diabetic (0), supporting early screening and helping reduce serious health risks.

---

## 📊 Dataset Description

- **Name:** Pima Indians Diabetes Database  
- **Total Samples:** 768 patients  
- **Population:** Adult female patients of Pima Indian heritage  
- **Features (8):**
  - Pregnancies
  - Glucose
  - Blood Pressure
  - Skin Thickness
  - Insulin
  - BMI (Body Mass Index)
  - Diabetes Pedigree Function
  - Age
- **Target Label:**
  - `Outcome` → (0 = Non-diabetic, 1 = Diabetic)

> Source: Public dataset available on Kaggle and UCI Machine Learning Repository

This dataset is widely used in the medical data science community due to its relevance and predictive value.

---

## 🧠 Methodology

### 1️⃣ Data Preprocessing
- Handling missing/illogical values (e.g., zero glucose or zero BMI)
- Replacing invalid values with median per feature
- **StandardScaler** applied for normalization
- Stratified **80/20 train-test split**

### 2️⃣ Exploratory Data Analysis
- Summary statistics and visualization of medical features
- Heatmap to examine correlation patterns
- Boxplots to observe distribution differences between diabetic vs non-diabetic groups

### 3️⃣ Machine Learning Models Used
| Model | Description |
|-------|-------------|
| Logistic Regression | Baseline linear classifier |
| Support Vector Machine (RBF) | Handles nonlinear patterns |
| Random Forest | Ensemble model with best performance |

### 4️⃣ Evaluation Metrics
- Accuracy  
- Precision  
- Recall  
- F1-score  
- ROC Curve  
- Confusion Matrix  
- Feature Importance Visualization  

---

## 🚀 Results

| Model | Accuracy | Precision | Recall | F1-score |
|------|:--------:|:---------:|:------:|:--------:|
| Logistic Regression | 70.78% | 60.00% | 50.00% | 54.55% |
| SVM (RBF) | 74.03% | 65.21% | 55.56% | 60.00% |
| **Random Forest** | **77.92%** | **71.73%** | **61.11%** | **66.00%** |

✔ Random Forest achieved the **best predictive performance**  
✔ Glucose & BMI found to be the **most influential predictors**  
✔ Results align with real medical evidence

---

## 📁 Repository Structure

```text
.
├── data/
│   └── diabetes.csv
├── notebooks/
│   └── diabetes_prediction.ipynb
├── figures/
│   ├── heatmap.png
│   ├── boxplots.png
│   ├── confusion_matrix_rf.png
│   ├── roc_curve.png
│   └── feature_importance.png
├── paper/
│   └── Diabetes_Prediction_IEEE_Paper.pdf
└── README.md


👩‍💻 Author
Nada Altalea
Department of Information Systems
King Khalid University, Saudi Arabia
