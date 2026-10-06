import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    classification_report
)


# 1. Load the dataset
data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")

print("FIRST 5 ROWS:")
print(X.head())

print("\nDATASET SHAPE:")
print(X.shape)

print("\nCLASS DISTRIBUTION:")
print(y.value_counts())


# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 3. Standardize the features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 4. Train Logistic Regression model
model = LogisticRegression(max_iter=1000)

model.fit(X_train_scaled, y_train)


# 5. Make predictions
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]


# 6. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nCONFUSION MATRIX:")
print(cm)


# 7. Precision
precision = precision_score(y_test, y_pred)

print("\nPRECISION:")
print(precision)


# 8. Recall
recall = recall_score(y_test, y_pred)

print("\nRECALL:")
print(recall)


# 9. ROC-AUC
roc_auc = roc_auc_score(y_test, y_prob)

print("\nROC-AUC:")
print(roc_auc)


# 10. Classification Report
print("\nCLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred))


# 11. Display Confusion Matrix
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=data.target_names
)

disp.plot()
plt.title("Confusion Matrix")
plt.show()


# 12. ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure()
plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()


# 13. Threshold tuning
threshold = 0.40

y_pred_threshold = (y_prob >= threshold).astype(int)

threshold_precision = precision_score(
    y_test,
    y_pred_threshold
)

threshold_recall = recall_score(
    y_test,
    y_pred_threshold
)

print("\nTHRESHOLD TUNING")
print("Threshold:", threshold)
print("Precision:", threshold_precision)
print("Recall:", threshold_recall)


# 14. Sigmoid function
import numpy as np

z = np.linspace(-10, 10, 100)

sigmoid = 1 / (1 + np.exp(-z))

plt.figure()
plt.plot(z, sigmoid)

plt.xlabel("z")
plt.ylabel("Sigmoid(z)")
plt.title("Sigmoid Function")
plt.grid()

plt.show()