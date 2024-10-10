from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.dummy import DummyRegressor

file_path = os.path.join(os.path.dirname(__file__), 'week3.csv')
column_names = ['Feature1', 'Feature2', 'Target']
data = pd.read_csv(file_path, names=column_names, skiprows=1)

X = data[['Feature1', 'Feature2']].values
y = data['Target'].values

poly = PolynomialFeatures(degree=5)
X_poly = poly.fit_transform(X)

baseline_regression = DummyRegressor(strategy='mean').fit(X_poly, y)
baseline_predictions = baseline_regression.predict(X_poly)
error = mean_squared_error(y, baseline_predictions)
print('Baseline Model')
print(f'Mean square error: {error}')

C_values = [1, 10, 100, 1000]

for C in C_values:
    model = Lasso(alpha=1/(2*C), max_iter=10000)
    model.fit(X_poly, y)
    
    print(f'C: {C}')
    print(f'Coefficients: {model.coef_}')
    print(f'Intercept: {model.intercept_}')
    error = mean_squared_error(y, model.predict(X_poly))
    print(f'Mean square error: {error}')
    print("\n")

grid = np.linspace(-5, 5, 50)
X_test = np.array([[i, j] for i in grid for j in grid])
X_test_poly = poly.transform(X_test)

fig, axes = plt.subplots(1, 4, subplot_kw={'projection': '3d'}, figsize=(24, 6))
colors = ['blue', 'green', 'orange', 'purple']

for ax, C, color in zip(axes, C_values, colors):
    model = Lasso(alpha=1/(2*C), max_iter=10000)
    model.fit(X_poly, y)
    predictions = model.predict(X_test_poly)
    predictions = predictions.reshape((50, 50))
    
    ax.plot_trisurf(X_test[:, 0], X_test[:, 1], predictions.flatten(), color=color, alpha=0.5)
    ax.scatter(X[:, 0], X[:, 1], y, color='red', label='Training Data')
    
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Target')
    ax.set_title(f'Lasso predictions C = {C}')
    
    ax.legend()

output_path = os.path.join(os.path.dirname(__file__), 'A(iii).png')
plt.savefig(output_path)
plt.show()
