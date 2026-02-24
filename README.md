# IBM HR Employee Attrition Analysis

> **Can we predict which employees are about to quit — before they hand in their resignation?**
>
> This project builds a full end-to-end machine learning pipeline on the IBM HR Analytics dataset to identify employees at high risk of attrition, explain *why* they are at risk, and provide actionable recommendations for HR teams.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Business Problem](#business-problem)
3. [Dataset](#dataset)
4. [Project Structure](#project-structure)
5. [Methodology](#methodology)
6. [Key Findings](#key-findings)
7. [Model Performance](#model-performance)
8. [SHAP Interpretability](#shap-interpretability)
9. [How to Run](#how-to-run)
10. [Tech Stack](#tech-stack)
11. [Results Summary](#results-summary)

---

## Project Overview

Employee attrition costs companies **50%–200% of an employee's annual salary** in recruitment, onboarding, and lost productivity. Yet most HR departments react *after* the resignation letter arrives.

This project takes a **proactive, data-driven approach**:
- Analyze 35 features across 1,470 IBM employees
- Engineer meaningful business signals beyond raw survey scores
- Train and compare multiple classifiers (Logistic Regression, Random Forest, XGBoost)
- Use SHAP values to explain *individual* attrition risk — not just aggregate feature importance

---

## Business Problem

| Metric | Value |
|--------|-------|
| Dataset attrition rate | 16.1% (237 / 1,470 employees) |
| Industry average attrition cost | 50–200% of annual salary |
| Target outcome | Binary classification: `Attrition = Yes / No` |

**Key questions this project answers:**
1. Which employees are most likely to leave in the next period?
2. What are the top drivers of attrition across the organization?
3. For a specific high-risk employee, what factors are pushing them toward the exit — and what can HR do about it?

---

## Dataset

**Source:** [IBM HR Analytics Employee Attrition & Performance — Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)

| Property | Detail |
|----------|--------|
| Rows | 1,470 employees |
| Columns | 35 features |
| Target | `Attrition` (Yes / No) |
| Class imbalance | ~84% No / ~16% Yes — addressed with SMOTE |

**Key feature categories:**

| Category | Features |
|----------|----------|
| Demographics | Age, Gender, MaritalStatus, Education, EducationField |
| Job Info | JobRole, Department, JobLevel, BusinessTravel |
| Compensation | MonthlyIncome, HourlyRate, StockOptionLevel, PercentSalaryHike |
| Satisfaction | JobSatisfaction, EnvironmentSatisfaction, RelationshipSatisfaction, WorkLifeBalance |
| Career | YearsAtCompany, YearsInCurrentRole, YearsSinceLastPromotion, NumCompaniesWorked |
| Work Pattern | OverTime, DistanceFromHome, TrainingTimesLastYear |

---

## Project Structure

```
ibm-hr-attrition/
│
├── ibm_hr_attrition_kaggle.ipynb   # Main analysis notebook (73 cells)
│
├── README.md                        # This file
│
└── (supporting data files referenced in notebook)
```

**Notebook sections:**

| Section | Content |
|---------|----------|
| Section 1 | Data loading, initial exploration, class imbalance diagnosis |
| Section 2 | Exploratory Data Analysis — static + interactive Plotly charts |
| Section 3 | Outlier detection (IQR + Z-score) |
| Section 4 | Feature engineering (8 engineered features) |
| Section 5 | Preprocessing — encoding, scaling, SMOTE oversampling |
| Section 6 | Model training — Logistic Regression, Random Forest, XGBoost |
| Section 7 | Model evaluation — ROC-AUC, Precision-Recall, confusion matrix |
| Section 8 | SHAP interpretability — global + local waterfall diagnosis |

---

## Methodology

### 1. Exploratory Data Analysis
- Attrition rate breakdown by JobRole, Department, MaritalStatus, OverTime
- Correlation heatmap to identify feature relationships
- Interactive Plotly violin & box plots for salary and distance distributions

### 2. Feature Engineering
Eight new features were engineered to capture business signals not present in raw data:

| Feature | Formula / Logic | Business Insight |
|---------|----------------|------------------|
| `Overtime_Satisfaction` | OverTime × JobSatisfaction | Captures overworked + unhappy compound risk |
| `Years_Per_Company` | TotalWorkingYears / (NumCompaniesWorked + 1) | Job hopper tendency |
| `Promotion_Lag` | YearsAtCompany − YearsSinceLastPromotion | Stagnation signal |
| `Compensation_Satisfaction` | MonthlyIncome × JobSatisfaction | Pay-happiness alignment |
| `Work_Life_Distance` | WorkLifeBalance × DistanceFromHome | Commute stress × balance |
| `Loyalty_Score` | YearsAtCompany × JobSatisfaction | Engagement depth |
| `Income_vs_Role_Avg` | MonthlyIncome / mean(MonthlyIncome by JobRole) | Below market pay signal |
| `Career_Stability` | TotalWorkingYears / (NumCompaniesWorked + 1) | Long-term stability index |

### 3. Handling Class Imbalance
The dataset is imbalanced (84% No / 16% Yes). **SMOTE** (Synthetic Minority Oversampling Technique) was applied on the training set only to avoid data leakage.

### 4. Models Trained

| Model | Notes |
|-------|-------|
| Logistic Regression | Baseline with L2 regularization |
| Random Forest | 200 estimators, class_weight='balanced' |
| XGBoost | scale_pos_weight tuned for imbalance, early stopping |

---

## Key Findings

**Top attrition risk factors (from SHAP analysis):**

1. **OverTime = Yes** — Single strongest predictor; employees working overtime are ~3x more likely to leave
2. **Low MonthlyIncome** — Below-role-average pay strongly correlates with attrition
3. **Low JobSatisfaction / EnvironmentSatisfaction** — Dissatisfied employees leave regardless of pay
4. **High DistanceFromHome** — Commute fatigue compounds dissatisfaction
5. **Single marital status** — More mobile, less constrained by family commitments
6. **Low StockOptionLevel** — Lack of long-term financial incentive reduces retention

**High-risk employee profile:**
- Works overtime
- Earns below the average for their role
- Low job and environment satisfaction
- Long commute
- Recently not promoted
- Single

---

## Model Performance

| Model | ROC-AUC | F1 (Attrition=Yes) | Recall (Attrition=Yes) | Precision (Attrition=Yes) |
|-------|---------|-------------------|----------------------|--------------------------|
| Logistic Regression | ~0.79 | ~0.52 | ~0.67 | ~0.43 |
| Random Forest | ~0.83 | ~0.58 | ~0.60 | ~0.56 |
| **XGBoost** | **~0.86** | **~0.63** | **~0.65** | **~0.61** |

> XGBoost achieved the best overall performance. For HR use cases, **Recall** is prioritized over Precision — it is more costly to miss a flight risk than to flag a false positive.

---

## SHAP Interpretability

This project goes beyond black-box prediction by using **SHAP (SHapley Additive exPlanations)** to explain model decisions at both global and individual levels.

### Global Feature Importance
SHAP summary plots reveal which features most influence attrition predictions across all employees, and in which direction (high/low values push toward Yes/No).

### Individual Waterfall Diagnosis
For any single high-risk employee, a **waterfall plot** breaks down exactly which factors are contributing to their attrition score:

```
Example output for Employee #142 (Predicted High Risk):
  Base rate:           0.16  (population average attrition)
  + OverTime=Yes:     +0.12
  + LowIncome:        +0.09
  + LowJobSat:        +0.07
  + HighDistance:     +0.05
  - HighLoyalty:      -0.03
  Final risk score:    0.46  → FLAG FOR HR REVIEW
```

This enables HR to have **specific, evidence-based conversations** with at-risk employees rather than generic retention programs.

---

## How to Run

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap plotly imbalanced-learn jupyter
```

### Steps
```bash
# 1. Clone the repository
git clone https://github.com/tom602229/ibm-hr-attrition.git
cd ibm-hr-attrition

# 2. Download the dataset from Kaggle
# https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset
# Place WA_Fn-UseC_-HR-Employee-Attrition.csv in the project root

# 3. Launch Jupyter
jupyter notebook ibm_hr_attrition_kaggle.ipynb

# 4. Run all cells (Kernel > Restart & Run All)
```

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.10+ |
| Data Manipulation | pandas, numpy |
| Visualization | matplotlib, seaborn, plotly |
| Machine Learning | scikit-learn, xgboost |
| Imbalanced Learning | imbalanced-learn (SMOTE) |
| Interpretability | shap |
| Environment | Jupyter Notebook |

---

## Results Summary

This project demonstrates a complete **production-ready ML pipeline** for HR attrition prediction:

- **Business framing** — Quantified attrition cost, defined success metrics aligned with HR goals
- **Rigorous EDA** — Both static and interactive visualizations reveal segment-level risk patterns
- **Feature engineering** — 8 domain-driven features that outperform raw survey scores
- **Proper ML hygiene** — SMOTE on train only, no data leakage, cross-validated evaluation
- **Model explainability** — SHAP waterfall plots enable individual-level HR intervention
- **Actionable output** — HR can prioritize outreach to flagged employees with specific talking points

> *"Predicting who will leave is only half the job. Knowing why they're leaving — and being able to act on it — is what makes this useful."*

---

## Author

**侯冠志 (Tom Hou)**
- GitHub: [@tom602229](https://github.com/tom602229)

---

*Dataset provided by IBM / Kaggle for educational purposes.*