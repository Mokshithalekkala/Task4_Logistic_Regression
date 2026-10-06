# Task 4: Classification with Logistic Regression

## Objective

The objective of this task is to build a binary classification model using Logistic Regression.

## Dataset

The Breast Cancer Wisconsin dataset from Scikit-learn is used for this task.

## Tools Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

## Steps Performed

1. Loaded the Breast Cancer Wisconsin dataset.
2. Split the dataset into training and testing data.
3. Standardized the features using StandardScaler.
4. Trained a Logistic Regression model.
5. Made predictions on the test data.
6. Evaluated the model using:
   - Confusion Matrix
   - Precision
   - Recall
   - ROC-AUC
7. Plotted the ROC curve.
8. Tuned the classification threshold.
9. Plotted and explained the sigmoid function.

## Conclusion

Logistic Regression is a classification algorithm that predicts the probability of a sample belonging to a particular class. The model performance can be evaluated using precision, recall, confusion matrix and ROC-AUC. Changing the classification threshold can affect the balance between precision and recall.