import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

class ModelRunner:
    def __init__(self, model_name, param_grid, use_pipeline=False):
        self.model_name = model_name
        self.param_grid = param_grid
        self.use_pipeline = use_pipeline
        self.grid_search = None
        self.final_model = None

    def perform_grid_search(self, X_train, y_train):
        if self.use_pipeline:
            pipeline = Pipeline([
                ('poly', PolynomialFeatures()),
                (self.model_name, LogisticRegression(max_iter=1000, penalty='l2') if self.model_name == 'log_reg' else KNeighborsClassifier())
            ])
            self.grid_search = GridSearchCV(pipeline, self.param_grid, cv=5, scoring='accuracy', return_train_score=True)
        else:
            model = LogisticRegression(max_iter=1000, penalty='l2') if self.model_name == 'log_reg' else KNeighborsClassifier()
            self.grid_search = GridSearchCV(model, self.param_grid, cv=5, scoring='accuracy', return_train_score=True)
        
        self.grid_search.fit(X_train, y_train)
        return self.grid_search

    def train_model(self, X_train, y_train):
        self.final_model = self.grid_search.best_estimator_
        self.final_model.fit(X_train, y_train)

    def evaluate_model(self, X_test, y_test):
        y_pred = self.final_model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        print(f"Confusion matrix for {self.model_name}:\n{cm}")
        y_prob = self.final_model.predict_proba(X_test)[:, 1] if hasattr(self.final_model, "predict_proba") else None
        return y_prob

    def plot_decision_boundary(self, X, y, index):
        xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100),
                             np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 100))
        Z = self.final_model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.figure(figsize=(10, 6))
        plt.contourf(xx, yy, Z, alpha=0.3)
        plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
        plt.title(f'Decision boundary of {self.model_name} (week4_{index}.csv)')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.savefig(f'Images/decision_boundary_week4_{index}_{self.model_name}.png')
        plt.close()

    def plot_roc_curve(self, y_test, y_prob, index):
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_score = auc(fpr, tpr)

        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, label=f'{self.model_name} (AUC = {auc_score:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve (week4_{index}.csv)')
        plt.legend(loc='best')
        plt.savefig(f'Images/roc_curve_week4_{index}.png')
        plt.close()

    def plot_cross_validation_results(self, index):
        results = self.grid_search.cv_results_
        mean_test_scores = results['mean_test_score']
        std_test_scores = results['std_test_score']
        
        plt.figure(figsize=(10, 6))
        if self.model_name == 'log_reg':
            poly_degrees = self.param_grid['poly__degree']
            C_values = self.param_grid['log_reg__C']
            for degree in poly_degrees:
                mask = results['param_poly__degree'] == degree
                plt.errorbar(C_values, mean_test_scores[mask], yerr=std_test_scores[mask], label=f'Degree {degree}', capsize=5)
            plt.xscale('log')
            plt.xlabel('C value (log scale)')
        else:
            k_values = self.param_grid['n_neighbors']
            plt.errorbar(k_values, mean_test_scores, yerr=std_test_scores, fmt='o-', capsize=5)
            plt.xlabel('Number of Neighbors (k)')
        
        plt.ylabel('Mean cross-validation accuracy')
        plt.title(f'Cross-validation accuracy for different parameters (week4_{index}.csv)')
        plt.legend()
        plt.savefig(f'Images/cross_validation_results_week4_{index}_{self.model_name}.png')
        plt.close()