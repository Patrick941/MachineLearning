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
import model_runner

os.makedirs('Images', exist_ok=True)
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

for index in range(1, 3):
    data = pd.read_csv(f'week4_{index}.csv', skiprows=1)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
    plt.title(f'Scatter plot of the data (week4_{index}.csv)')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    i_string = 'i' * index
    plt.savefig(f'Images/{i_string}(a(1)).png')
    plt.close()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    log_reg_runner = model_runner.ModelRunner('log_reg', X_train, y_train, X_test, y_test, i_string=i_string)
    log_reg_runner.train_model()
    log_reg_runner.predict()
    log_reg_runner.plot_confusion_matrix()
    log_reg_runner.plot_decision_boundary(index)
    log_reg_runner.plot_roc_curve(index)

    knn_runner = model_runner.ModelRunner('knn', X_train, y_train, X_test, y_test, i_string=i_string)
    knn_runner.train_model()
    knn_runner.predict()
    knn_runner.plot_confusion_matrix()
    knn_runner.plot_decision_boundary(index)
    knn_runner.plot_roc_curve(index)

    baseline_runner = model_runner.ModelRunner('baseline_most_frequent', X_train, y_train, X_test, y_test, i_string=i_string)
    baseline_runner.train_model()
    baseline_runner.predict()
    baseline_runner.plot_confusion_matrix()
    baseline_runner.plot_decision_boundary(index)
    baseline_runner.plot_roc_curve(index)
    
    baseline_runner = model_runner.ModelRunner('baseline_random', X_train, y_train, X_test, y_test, i_string=i_string)
    baseline_runner.train_model()
    baseline_runner.predict()
    baseline_runner.plot_confusion_matrix()
    baseline_runner.plot_decision_boundary(index)
    baseline_runner.plot_roc_curve(index)

