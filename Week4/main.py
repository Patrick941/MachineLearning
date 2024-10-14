import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import os

# Ensure the Images directory exists
os.makedirs('Images', exist_ok=True)

# Load the dataset from week4.csv, skipping the first row
data = pd.read_csv('week4_1.csv', skiprows=1)
X = data.iloc[:, :-1].values  # Assuming the last column is the target
y = data.iloc[:, -1].values

# Check for and handle missing values in the target variable
if np.any(pd.isnull(y)):
    raise ValueError("The target variable y contains NaN values. Please clean the data before proceeding.")

# Define the range of polynomial orders and C values to consider
poly_orders = [1, 2, 3, 4, 5]
C_values = np.logspace(-4, 4, 10)

# Create a pipeline with PolynomialFeatures and LogisticRegression
pipeline = Pipeline([
    ('poly', PolynomialFeatures()),
    ('logreg', LogisticRegression(penalty='l2', solver='liblinear'))
])

# Define the parameter grid for GridSearchCV
param_grid = {
    'poly__degree': poly_orders,
    'logreg__C': C_values
}

# Perform GridSearchCV to find the best parameters
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', return_train_score=True)
grid_search.fit(X, y)

# Extract the best parameters and the corresponding score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print(f"Best parameters: {best_params}")
print(f"Best cross-validation score: {best_score:.4f}")

# Plotting cross-validation results
results = grid_search.cv_results_
mean_test_scores = results['mean_test_score']
std_test_scores = results['std_test_score']

plt.figure(figsize=(10, 6))
for i, degree in enumerate(poly_orders):
    mask = results['param_poly__degree'] == degree
    plt.errorbar(C_values, mean_test_scores[mask], yerr=std_test_scores[mask], label=f'Degree {degree}')

plt.xscale('log')
plt.xlabel('C value (log scale)')
plt.ylabel('Mean cross-validation accuracy')
plt.title('Cross-validation accuracy for different polynomial degrees and C values')
plt.legend()
plt.savefig('Images/cross_validation_results.png')
plt.close()

# Train the final model with the best parameters
final_model = grid_search.best_estimator_
final_model.fit(X, y)

# Plot the decision boundary
xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100),
                     np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 100))
Z = final_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
plt.title('Decision boundary of the best model')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.savefig('Images/decision_boundary.png')
plt.close()