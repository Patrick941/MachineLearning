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
import knn
import logistic_regression

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
    plt.savefig(f'Images/scatter_plot_week4_{index}.png')
    plt.close()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    logistic_regression.run_logistic_regression(X_train, X_test, y_train, y_test, index)

    knn.run_knn(X_train, X_test, y_train, y_test, X, y, index)

    run_baseline(y_train, y_test, index)
