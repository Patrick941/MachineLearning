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

X1 = df.iloc[:, 0]
X2 = df.iloc[:, 1]
X = np.column_stack((X1, X2))
y = df.iloc[:, 2]

# Transform features to polynomial features of degree 2
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

plt.figure(figsize=(8, 6))
plt.scatter(X[y == 1, 0], X[y == 1, 1], marker='+', color='blue', label='+1')
plt.scatter(X[y == -1, 0], X[y == -1, 1], marker='o', color='red', label='-1')

plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend()
plt.title('2D Plot of Features with Target Values')
plt.grid(True)
# plt.show()

model = LogisticRegression()
model.fit(X_poly, y)

predictions = model.predict(X_poly)
coefficients = model.coef_[0]
intercept = model.intercept_[0]

print("Model coefficients:", coefficients)
print("Model intercept:", intercept)

predictions = model.predict(X_poly)
x_min, x_max = -1, 1
y_min, y_max = -1, 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

Z = poly.transform(np.c_[xx.ravel(), yy.ravel()])
Z = model.predict(Z)
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)
plt.scatter(X[y == 1, 0], X[y == 1, 1], marker='+', color='blue', label='+1')
plt.scatter(X[y == -1, 0], X[y == -1, 1], marker='o', color='red', label='-1')

plt.xlabel('x_1')
plt.ylabel('x_2')
plt.title('Decision Boundary with Training Data')
plt.grid(True)
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.show()

breakpoint