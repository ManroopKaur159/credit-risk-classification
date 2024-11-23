# Credit Risk Analysis Report

# Overview of the Analysis- The purpose of this analysis is to evaluate the performance of a machine learning model in predicting the risk of loan default.
This analysis aims to determine whether a given loan is healthy (label 0) or at high risk of defaulting (label 1).
The lending_data.csv file was read into a Pandas DataFrame.

# Financial Data
- Dataset: lending_data.csv
- Pandas DataFrame: lending_df
- Columns in lending_df: loan_size,	interest_rate,	borrower_income,	debt_to_income,	num_of_accounts,	derogatory_marks,	total_debt and	loan_status.
Value 0: Healthy loans
Value 1: High-risk loans
- Target Variable: loan_status
- Features: All remaining columns in the dataset were used to predict loan_status.

Information about variables we were trying to predict-
Value counts of loan_status in lending_df was
0    75036
1     2500
O means helathy loans and 1 means loans at the high risk of defaulting

# Stages of Machine Learning Process for this analysis-
During data preparation the lending_data.csv was read into a Pandas Dataframe.
- The target variable (y) was created from the loan_status column, where 0 indicates a healthy loan and 1 indicates a high-risk loan.
- The features (X) were created from the remaining columns of the dataset.

# Data Splitting
The dataset was split into training and testing sets using train_test_split where a random_state of 1 was set which gave the following results:
Training set: 58,152 records.
Testing set: 19,384 records.

#The Logistic Regression Model
A logistic regression model was used with the following parameters:
- Solver: lbfgs
- Maximum Iterations: 200
- Randomn State: 1
This model was trained using the training datasets x_train, y_train.
The model.predict(X_test) function was used to generate predictions for the testing dataset and stored the results in y_predictions.

# Confusion Matrix
A confusion matrix was created to visualize true positives, true negatives, false positives, and false negatives.
Evaluation: The matrix contains:
- True Negatives (18658): Correct predictions for healthy loans (0).
- False Positives (107): Incorrect predictions of high-risk loans (1) for healthy loans (0).
- False Negatives (37): Incorrect predictions of healthy loans (0) for high-risk loans (1).
- True Positives (582): Correct predictions for high-risk loans (1).

The confusion matrix results indicate that the logistic regression model performed well for predicting healthy loans (0) 
but could be improved for predicting high-risk loans (1).

# Classification Report
A classification report to assess accuracy, precision, recall, and F1-score for both loan statuses.
The logistic regression model predicts 0 which is the healthy loan very well, with 1.00 (100%) precision and nearly perfect recall of 0.99 (99%)
For 1 which is a high-risk loan, the model does not perform as strongly, achieving an 0.84 (84%) for precision and 0.94 (94%) for recall, which reflects a drop in precision and recall.

# Summary and Results
This model is well-suited for applications where predicting healthy loans is crucial, but changes may be needed if identifying high-risk loans with higher precision is a priority.
Overall, the model did achieve high accuracy and is reliable for predicting both labels, though there is room for improvement in identifying high-risk loans.
