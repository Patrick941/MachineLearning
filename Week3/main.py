from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline

file_path = os.path.join(os.path.dirname(__file__), 'week3.csv')

column_names = ['Feature1', 'Feature2', 'Target']
data = pd.read_csv(file_path, names=column_names, skiprows=1)

X = data[['Feature1', 'Feature2']].values
y = data['Target'].values

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], y)

ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Target')

# plt.show()
output_path = os.path.join(os.path.dirname(__file__), 'A(i).png')
plt.savefig(output_path)

poly = PolynomialFeatures(degree=5)
X_poly = poly.fit_transform(X)

C_values = [1, 10, 1000]

for C in C_values:
    lasso = Lasso(alpha=1.0 / C, max_iter=10000)
    lasso.fit(X_poly, y)
    

    print(f"Parameters for C={C}:")
    for feature, coef in zip(poly.get_feature_names_out(['Feature1', 'Feature2']), lasso.coef_):
        print(f"{feature}: {coef}")
    print("\n")
    
grid = np.linspace(-5, 5, 50)
X_test = np.array([[i, j] for i in grid for j in grid])

X_test_poly = poly.transform(X_test)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X[:, 0], X[:, 1], y, color='red', label='Training data')

for C in C_values:
    lasso = Lasso(alpha=1.0 / C, max_iter=10000)
    lasso.fit(X_poly, y)
    

    y_pred = lasso.predict(X_test_poly)
    

    y_pred = y_pred.reshape((50, 50))
    

    ax.plot_surface(grid, grid, y_pred, alpha=0.3, label=f'Predictions for C={C}')

ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Target')
ax.legend()

output_path = os.path.join(os.path.dirname(__file__), 'A(ii).png')
plt.savefig(output_path)
plt.show()