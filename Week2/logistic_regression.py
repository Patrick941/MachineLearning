import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

fileDirectory = os.path.dirname(os.path.abspath(__file__))
## Assumes only one csv
csv_file = [f for f in os.listdir(fileDirectory) if f.endswith('.csv')][0]
csv_path = os.path.join(fileDirectory, csv_file)
df = pd.read_csv(csv_path, skiprows=1)

# Extract features and target
X1 = df.iloc[:, 0]
X2 = df.iloc[:, 1]
X = np.column_stack((X1, X2))
y = df.iloc[:, 2]

model = LogisticRegression()
model.fit(X, y)

coefficients = model.coef_[0]
intercept = model.intercept_[0]
print("Model coefficients:", coefficients)
print("Model intercept:", intercept)
predictions = model.predict(X)
plt.figure(figsize=(8, 6))
plt.scatter(X[y == 1, 0], X[y == 1, 1], marker='o', color='blue', label='Actual +1')
plt.scatter(X[y == -1, 0], X[y == -1, 1], marker='o', color='red', label='Actual -1')
plt.scatter(X[predictions == 1, 0], X[predictions == 1, 1], marker='x', color='green', label='Predicted +1')
plt.scatter(X[predictions == -1, 0], X[predictions == -1, 1], marker='+', color='orange', label='Predicted -1')
x_values = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
decision_boundary = -(coefficients[0] * x_values + intercept) / coefficients[1]
plt.plot(x_values, decision_boundary, color='black', linestyle='--', label='Decision Boundary')

plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend()
plt.title('Logistic Regression Classifier')
plt.grid(True)
plt.savefig(os.path.join(fileDirectory, 'A(iii).png'))

X_squared = np.column_stack((X, X**2))
model = LogisticRegression()
model.fit(X_squared, y)

predictions = model.predict(X_squared)
coefficients = model.coef_[0]
intercept = model.intercept_[0]

print("Model coefficients:", coefficients)
print("Model intercept:", intercept)

baseline_model = DummyClassifier(strategy='most_frequent')
baseline_model.fit(X_squared, y)
baseline_predictions = baseline_model.predict(X_squared)
baseline_accuracy = accuracy_score(y, baseline_predictions)

logistic_accuracy = accuracy_score(y, predictions)

print("Baseline model accuracy:", baseline_accuracy)
print("Logistic regression model accuracy:", logistic_accuracy)

plt.figure(figsize=(8, 6))
plt.scatter(X[y == 1, 0], X[y == 1, 1], marker='o', color='blue', label='Actual +1')
plt.scatter(X[y == -1, 0], X[y == -1, 1], marker='o', color='red', label='Actual -1')
plt.scatter(X[predictions == 1, 0], X[predictions == 1, 1], marker='x', color='green', label='Predicted +1')
plt.scatter(X[predictions == -1, 0], X[predictions == -1, 1], marker='+', color='orange', label='Predicted -1')

plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend()
plt.title('Logistic Regression with Squared Features')
plt.grid(True)
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.savefig(os.path.join(fileDirectory, 'C(i).png'))

x1_sorted = np.sort(X[:, 0])
x2_sorted = np.sort(X[:, 1])

y_values = (-model.coef_[0][0] * x1_sorted - model.coef_[0][2] * x1_sorted**2 - model.intercept_[0]) / (model.coef_[0][1] + model.coef_[0][3] * x2_sorted)

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))

Z = model.predict(np.column_stack((xx.ravel(), yy.ravel(), (xx**2).ravel(), (yy**2).ravel())))
Z = Z.reshape(xx.shape)

plt.plot(x1_sorted, y_values, 'k-')
plt.title('Logistic Regression Classifier with Squared Features')
plt.savefig(os.path.join(fileDirectory, 'C(iv).png'))