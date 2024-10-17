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
import model_runner as model_runner

def run_baseline(y_train, y_test, index):
    most_frequent_class = Counter(y_train).most_common(1)[0][0]
    baseline_most_frequent = np.full_like(y_test, most_frequent_class)
    np.random.seed(42)
    baseline_random = np.random.choice(np.unique(y_train), size=y_test.shape)
    cm_baseline_most_frequent = confusion_matrix(y_test, baseline_most_frequent)
    cm_baseline_random = confusion_matrix(y_test, baseline_random)
    print(f"Confusion matrix for baseline most frequent (week4_{index}.csv):\n{cm_baseline_most_frequent}")
    print(f"Confusion matrix for baseline random (week4_{index}.csv):\n{cm_baseline_random}")

os.makedirs('Images', exist_ok=True)
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

for index in range(1, 3):
    data = pd.read_csv(f'week4_{index}.csv', skiprows=1)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    if np.any(pd.isnull(y)):
        raise ValueError("The target variable y contains NaN values. Please clean the data before proceeding.")

    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
    plt.title(f'Scatter plot of the data (week4_{index}.csv)')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    i_string = 'i' * index
    plt.savefig(f'Images/{i_string}(a(1)).png')
    plt.close()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    log_reg_param_grid = {'poly__degree': list(range(1, 6)), 'log_reg__C': np.logspace(-3, 3, 7)}
    log_reg_runner = model_runner.ModelRunner('log_reg', log_reg_param_grid, use_pipeline=True, i_string=i_string)

    log_reg_runner.perform_grid_search(X_train, y_train)
    log_reg_runner.train_model(X_train, y_train)
    y_prob_log_reg = log_reg_runner.evaluate_model(X_test, y_test)

    log_reg_runner.plot_decision_boundary(X_train, y_train, index)
    log_reg_runner.plot_cross_validation_results(index)

    log_reg_runner.plot_roc_curve(y_test, y_prob_log_reg, index)
        
    log_reg_runner.plot_confusion_matrix(y_test, index)

    knn_param_grid = {'n_neighbors': list(range(1, 21))}
    knn_runner = model_runner.ModelRunner('knn', knn_param_grid, i_string=i_string)

    knn_runner.perform_grid_search(X_train, y_train)
    knn_runner.train_model(X_train, y_train)
    y_prob_knn = knn_runner.evaluate_model(X_test, y_test)

    knn_runner.plot_decision_boundary(X_train, y_train, index)
    knn_runner.plot_cross_validation_results(index)

    knn_runner.plot_roc_curve(y_test, y_prob_knn, index)
        
    knn_runner.plot_confusion_matrix(y_test, index)

    run_baseline(y_train, y_test, index)
