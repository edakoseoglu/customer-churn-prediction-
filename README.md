# Customer Subscription Prediction with Machine Learning

This project applies **predictive analytics** to historical customer data from a subscription-based newspaper company. The primary objective was to build, compare, and evaluate machine learning models that predict whether a potential customer will **subscribe** or **not subscribe**.  

---

## Project Overview
Subscription-based businesses depend on customer acquisition and retention for sustainable revenue. This project uses customer demographics, pricing, and engagement features to forecast subscription decisions.

- **Dataset**: Kaggle – [Newspaper Churn](https://www.kaggle.com/datasets/andieminogue/newspaper-churn/data)  
- **Target Variable**: `Subscriber` (`YES` / `NO`)  
- **Key Features**:  
  - **Demographics**: age range, ethnicity, household income  
  - **Behavioural**: source channel, language, delivery period  
  - **Pricing**: weekly fee  
  - **Engagement**: reward program participation, tenure  

---

## Methods & Workflow
1. **Data Preprocessing**
   - Missing value imputation (mode for categorical, etc.)
   - Encoding strategies: one-hot, ordinal, binary
   - Feature scaling for numerical variables (e.g., income, weekly fee)
   - Stratified train-test split (80/20) to handle class imbalance  

2. **Exploratory Data Analysis (EDA)**
   - Distribution analysis of target variable (imbalanced dataset)
   - Correlation analysis of numeric features
   - Visualisation of missing data and feature distributions  

3. **Model Development**
   - **Logistic Regression** (baseline)
   - **Random Forest** (feature importance analysis)
   - **XGBoost** (boosted trees with hyperparameter tuning)
   - **Neural Network** (tested for complex patterns)

4. **Evaluation**
   - Metrics: Accuracy, Precision, Recall, F1-score (focus on minority class recall)
   - Threshold tuning (precision-recall trade-off)
   - Train vs Test comparisons to detect overfitting  

---

## Results
- **Logistic Regression**: F1 ≈ 0.27 – baseline, poor recall for subscribers  
- **Random Forest**: F1 ≈ 0.50 – better at capturing non-linear relationships  
- **Neural Network**: F1 ≈ 0.45 – high risk of overfitting with small dataset  
- **XGBoost (Final Model)**: **F1 ≈ 0.60** – best trade-off after hyperparameter tuning and threshold optimisation  

**Key Business Insight:**  
XGBoost provided the best ability to identify potential subscribers, balancing recall and precision
