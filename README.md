Credit Risk Analysis Report
Overview of the Analysis
The purpose of this analysis is to evaluate the performance of a machine learning model in predicting the risk of loan default.
This analysis aims to determine whether a given loan is:
Healthy (label 0)
At high risk of defaulting (label 1)
The lending_data.csv file was read into a Pandas DataFrame.
Financial Data
Dataset: lending_data.csv
Pandas DataFrame: lending_df
Columns in lending_df:
loan_size, interest_rate, borrower_income, debt_to_income, num_of_accounts, derogatory_marks, total_debt, and loan_status.
Target Variable: loan_status
Value 0: Healthy loans
Value 1: High-risk loans
Features: All remaining columns in the dataset were used to predict loan_status.
Information about Variables
Value counts of loan_status:
0 (Healthy Loans): 75,036
1 (High-Risk Loans): 2,500
Interpretation:
0 represents healthy loans.
1 represents loans at a high risk of defaulting.
Stages of Machine Learning Process
Data Preparation
The lending_data.csv file was read into a Pandas DataFrame.
Target Variable (y): Created from the loan_status column.
0: Healthy loans
1: High-risk loans
Features (X): Created from the remaining columns in the dataset.
Data Splitting
The dataset was split into training and testing sets using train_test_split with:
Training set: 58,152 records.
Testing set: 19,384 records.
Random State: 1 (ensures reproducibility).
Logistic Regression Model
Model Parameters:
Solver: lbfgs
Maximum Iterations: 200
Random State: 1
Training: Model was trained using the training datasets (x_train, y_train).
Prediction: Predictions for the testing dataset (X_test) were generated using model.predict(X_test) and stored in y_predictions.
Confusion Matrix
Evaluation of Results:
True Negatives (18,658): Correct predictions for healthy loans (0).
False Positives (107): Incorrect predictions of high-risk loans (1) for healthy loans (0).
False Negatives (37): Incorrect predictions of healthy loans (0) for high-risk loans (1).
True Positives (582): Correct predictions for high-risk loans (1).
Interpretation:
The model performs very well for predicting healthy loans (0) but shows some weaknesses in identifying high-risk loans (1).
Classification Report
Class 0 (Healthy Loans):
Precision: 1.00 (100%)
Recall: 0.99 (99%)
F1-Score: 1.00 (100%)
Class 1 (High-Risk Loans):
Precision: 0.84 (84%)
Recall: 0.94 (94%)
F1-Score: 0.89 (89%)
Overall Metrics:
Accuracy: 0.99 (99%)
Weighted Precision, Recall, F1-Score: 0.99
Summary and Results
Strengths:
Excellent performance for predicting healthy loans (0) with perfect precision and near-perfect recall.
High overall accuracy (99%).
Weaknesses:
Lower precision (84%) for high-risk loans (1), leading to false positives.
Some missed high-risk loans due to false negatives.
Recommendation:
This model is ideal for applications focused on predicting healthy loans accurately.
Improvements are recommended if higher precision for high-risk loans is required.
