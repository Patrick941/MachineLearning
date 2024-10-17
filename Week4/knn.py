import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import os
from collections import Counter
from sklearn.pipeline import Pipeline

def run_knn(X_train, X_test, y_train, y_test, X, y, index):
    k_values = list(range(1, 21))
    knn = KNeighborsClassifier()
    param_grid = {
        'n_neighbors': k_values
    }
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', return_train_score=True)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    print(f"Best parameters for kNN (week4_{index}.csv): {best_params}")
    print(f"Best cross-validation score for kNN (week4_{index}.csv): {best_score:.4f}")
    final_model = grid_search.best_estimator_
    final_model.fit(X_train, y_train)
    y_pred_knn = final_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_knn)
    print(f"Confusion matrix for kNN (week4_{index}.csv):\n{cm}")
    plot_cross_validation_results(grid_search, {'n_neighbors': list(range(1, 21))}, index, 'knn')
    plot_decision_boundary(X, y, final_model, index)
    y_prob = final_model.predict_proba(X_test)[:, 1]
    return y_prob

def perform_grid_search_knn(X_train, y_train):
    

def plot_cross_validation_results(grid_search, param_grid, index, model_name):
    results = grid_search.cv_results_
    mean_test_scores = results['mean_test_score']
    std_test_scores = results['std_test_score']
    plt.figure(figsize=(10, 6))
    if model_name == 'log_reg':
        poly_degrees = param_grid['poly__degree']
        C_values = param_grid['log_reg__C']
        for degree in poly_degrees:
            mask = results['param_poly__degree'] == degree
            plt.errorbar(C_values, mean_test_scores[mask], yerr=std_test_scores[mask], label=f'Degree {degree}', capsize=5)
        plt.xscale('log')
        plt.xlabel('C value (log scale)')
    else:
        k_values = param_grid['n_neighbors']
        plt.errorbar(k_values, mean_test_scores, yerr=std_test_scores, fmt='o-', capsize=5)
        plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Mean cross-validation accuracy')
    plt.title(f'Cross-validation accuracy for different parameters (week4_{index}.csv)')
    plt.legend()
    plt.savefig(f'Images/cross_validation_results_week4_{index}_{model_name}.png')
    plt.close()
    
def plot_decision_boundary(X, y, model, index):
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100),
                         np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 100))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
    plt.title(f'Decision boundary of the best kNN model (week4_{index}.csv)')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.savefig(f'Images/decision_boundary_week4_{index}_knn.png')
    plt.close()
