import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
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
plt.title('2D Plot of Features with Target Values and Predictions')
plt.grid(True)
plt.savefig(os.path.join(fileDirectory, 'A(iii).png'))

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
model = LogisticRegression()
model.fit(X, y)

predictions = model.predict(X)
coefficients = model.coef_[0]
intercept = model.intercept_[0]

print("Model coefficients:", coefficients)
print("Model intercept:", intercept)

model_polynomial = LogisticRegression()
model_polynomial.fit(X_poly, y)
predictions = model_polynomial.predict(X_poly)
x_min, x_max = -1, 1
y_min, y_max = -1, 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

Z = poly.transform(np.c_[xx.ravel(), yy.ravel()])
Z = model_polynomial.predict(Z)
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)
plt.scatter(X[y == 1, 0], X[y == 1, 1], marker='o', color='blue', label='Actual +1')
plt.scatter(X[y == -1, 0], X[y == -1, 1], marker='o', color='red', label='Actual -1')
plt.scatter(X[predictions == 1, 0], X[predictions == 1, 1], marker='x', color='green', label='Predicted +1')
plt.scatter(X[predictions == -1, 0], X[predictions == -1, 1], marker='+', color='orange', label='Predicted -1')

plt.xlabel('x_1')
plt.ylabel('x_2')
plt.title('Decision Boundary with Training Data')
plt.grid(True)
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.savefig(os.path.join(fileDirectory, 'C(i).png'))