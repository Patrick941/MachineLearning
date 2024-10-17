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

def run_logistic_regression(X_train, X_test, y_train, y_test, index):
    # Perform grid search to find the best hyperparameters
    grid_search_result = grid_search(X_train, y_train)
    best_params = grid_search_result.best_params_
    best_score = grid_search_result.best_score_
    
    print(f"Best parameters for Logistic Regression (week4_{index}.csv): {best_params}")
    print(f"Best cross-validation score for Logistic Regression (week4_{index}.csv): {best_score:.4f}")
    
    # Train the final model with the best parameters
    final_model = grid_search_result.best_estimator_
    final_model.fit(X_train, y_train)
    
    # Determine the power for plotting decision boundaries
    best_power = int(np.log10(best_params['log_reg__C']) + 3)
    
    # Plot decision boundaries
    decision_boundary(X_train, y_train, final_model, index, best_params['log_reg__C'], best_power)
    
    # Predict and evaluate the model on the test set
    y_pred = final_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion matrix for Logistic Regression (week4_{index}.csv):\n{cm}")
    
    # Plot cross-validation results
    plot_cross_validation(grid_search_result, {'poly__degree': list(range(1, 6)), 'log_reg__C': np.logspace(-3, 3, 7)}, index, 'log_reg')
    
    # Predict probabilities for ROC curve
    y_prob = final_model.predict_proba(X_test)[:, 1]
    return y_prob

def plot_roc_curve(y_test, y_prob, baseline_most_frequent, baseline_random, index, model_name):
    # Compute ROC curve and AUC for the model and baselines
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fpr_baseline_most_frequent, tpr_baseline_most_frequent, _ = roc_curve(y_test, baseline_most_frequent)
    fpr_baseline_random, tpr_baseline_random, _ = roc_curve(y_test, baseline_random)
    
    auc_score = auc(fpr, tpr)
    auc_baseline_most_frequent = auc(fpr_baseline_most_frequent, tpr_baseline_most_frequent)
    auc_baseline_random = auc(fpr_baseline_random, tpr_baseline_random)
    
    # Plot ROC curve
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.2f})')
    plt.scatter(fpr_baseline_most_frequent, tpr_baseline_most_frequent, label=f'Baseline Most Frequent (AUC = {auc_baseline_most_frequent:.2f})', color='red')
    plt.scatter(fpr_baseline_random, tpr_baseline_random, label=f'Baseline Random (AUC = {auc_baseline_random:.2f})', color='green')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve (week4_{index}.csv)')
    plt.legend(loc='best')
    plt.savefig(f'Images/roc_curve_week4_{index}.png')
    plt.close()

def decision_boundary(X, y, model, index, C_value, power):
    # Create a mesh grid for plotting decision boundaries
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100),
                         np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 100))
    
    plt.figure(figsize=(15, 10))
    for i, C_value in enumerate(np.logspace(-3, 3, power), 1):
        model = LogisticRegression(C=C_value, max_iter=1000, penalty='l2')
        model.fit(X, y)
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        plt.subplot(3, 3, i)
        plt.contourf(xx, yy, Z, alpha=0.3)
        plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
        plt.title(f'C={C_value}')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
    
    plt.suptitle(f'Decision boundaries of Logistic Regression for different C values (week4_{index}.csv)')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'Images/decision_boundary_week4_{index}_log_reg_C_values.png')
    plt.close()

def grid_search(X_train, y_train):
    # Define the parameter grid for grid search
    poly_degrees = list(range(1, 6))
    C_values = np.logspace(-3, 3, 7)
    
    # Create a pipeline with polynomial features and logistic regression
    pipeline = Pipeline([
        ('poly', PolynomialFeatures()),
        ('log_reg', LogisticRegression(max_iter=1000, penalty='l2'))
    ])
    
    param_grid = {
        'poly__degree': poly_degrees,
        'log_reg__C': C_values
    }
    
    # Perform grid search with cross-validation
    grid_search_result = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', return_train_score=True)
    grid_search_result.fit(X_train, y_train)
    return grid_search_result

def plot_cross_validation(grid_search, param_grid, index, model_name):
    # Extract cross-validation results
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