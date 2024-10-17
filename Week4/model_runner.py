import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

class ModelRunner:
    def __init__(self, model_name, X_train, y_train, X_test, y_test, i_string=''):
        self.model_name = model_name
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.i_string = i_string
        self.final_model = None
        self.best_params = None
        self.y_pred = None
        self.y_prob = None

    def train_model(self):
        best_score = float('inf')
        best_model = None
        best_params = None
        k_values = list(range(1, 41))
        C_values = np.logspace(-3, 3, 7)
        poly_degrees = list(range(1, 6))
        
        cv_results = []

        improvement_threshold = 0.01 

        if self.model_name != 'log_reg':
            poly_degrees = [1]
        for poly_degree in poly_degrees:
            poly = PolynomialFeatures(degree=poly_degree)
            X_poly = poly.fit_transform(self.X_train)

            if self.model_name == 'log_reg':
                for C_value in C_values:
                    model = LogisticRegression(C=C_value)
                    scores = cross_val_score(model, X_poly, self.y_train, cv=5, scoring='neg_mean_squared_error')
                    mean_score = -scores.mean()
                    cv_results.append((poly_degree, C_value, mean_score))

                    if best_score - mean_score > improvement_threshold:
                        best_score = mean_score
                        best_model = model
                        best_params = {'poly_degree': poly_degree, 'C': C_value}
            elif self.model_name == 'knn':
                for k_value in k_values:
                    model = KNeighborsClassifier(n_neighbors=k_value)
                    scores = cross_val_score(model, X_poly, self.y_train, cv=5, scoring='neg_mean_squared_error')
                    mean_score = -scores.mean()
                    cv_results.append((poly_degree, k_value, mean_score))

                    if best_score - mean_score > improvement_threshold:
                        best_score = mean_score
                        best_model = model
                        best_params = {'poly_degree': poly_degree, 'n_neighbors': k_value}
            elif self.model_name == 'baseline_most_frequent':
                model = DummyClassifier(strategy='most_frequent')
                scores = cross_val_score(model, X_poly, self.y_train, cv=5, scoring='neg_mean_squared_error')
                mean_score = -scores.mean()
                cv_results.append((poly_degree, 'most_frequent', mean_score))

                if best_score - mean_score > improvement_threshold:
                    best_score = mean_score
                    best_model = model
                    best_params = {'poly_degree': poly_degree, 'strategy': 'most_frequent'}
            elif self.model_name == 'baseline_random':
                model = DummyClassifier(strategy='uniform')
                scores = cross_val_score(model, X_poly, self.y_train, cv=5, scoring='neg_mean_squared_error')
                mean_score = -scores.mean()
                cv_results.append((poly_degree, 'random', mean_score))

                if best_score - mean_score > improvement_threshold:
                    best_score = mean_score
                    best_model = model
                    best_params = {'poly_degree': poly_degree, 'strategy': 'uniform'}

        plt.figure(figsize=(12, 8))
        for poly_degree in poly_degrees:
            if self.model_name == 'log_reg':
                scores = [result[2] for result in cv_results if result[0] == poly_degree]
                plt.plot(C_values, scores, label=f'Poly Degree {poly_degree}')
            elif self.model_name == 'knn':
                scores = [result[2] for result in cv_results if result[0] == poly_degree]
                plt.plot(k_values, scores, label=f'Poly Degree {poly_degree}')
            elif self.model_name in ['baseline_most_frequent', 'baseline_random']:
                scores = [result[2] for result in cv_results if result[0] == poly_degree]
                plt.plot([0], scores, label=f'Poly Degree {poly_degree}', marker='o')

        plt.xlabel('C values' if self.model_name == 'log_reg' else 'k values' if self.model_name == 'knn' else 'Baseline')
        plt.ylabel('Mean Squared Error')
        plt.title(f'Cross-Validation Results for {self.model_name}')
        plt.legend()
        plt.xscale('log' if self.model_name == 'log_reg' else 'linear')
        if self.model_name == 'log_reg':
            plt.savefig(f'Images/{self.i_string}(a(2)).png')
        elif self.model_name == 'knn':
            plt.savefig(f'Images/{self.i_string}(b(2)).png')
        plt.close()

        self.best_params = best_params
        self.poly = PolynomialFeatures(degree=best_params['poly_degree'])
        X_poly_train = self.poly.fit_transform(self.X_train)
        self.final_model = best_model.fit(X_poly_train, self.y_train)

    def predict(self):
        poly = PolynomialFeatures(degree=self.best_params['poly_degree'])
        X_poly_test = poly.fit_transform(self.X_test)
        self.y_pred = self.final_model.predict(X_poly_test)
        self.y_prob = self.final_model.predict_proba(X_poly_test)[:, 1]

    def plot_confusion_matrix(self):
        cm = confusion_matrix(self.y_test, self.y_pred)
        
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Confusion matrix for {self.model_name}')
        plt.colorbar()
        tick_marks = np.arange(len(np.unique(self.y_test)))
        plt.xticks(tick_marks, tick_marks)
        plt.yticks(tick_marks, tick_marks)
        
        fmt = 'd'
        thresh = cm.max() / 2.
        for i, j in np.ndindex(cm.shape):
            plt.text(j, i, format(cm[i, j], fmt),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
        
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        if self.model_name == 'log_reg':
            plt.savefig(f'Images/{self.i_string}(c(1)).png')
        elif self.model_name == 'knn':
            plt.savefig(f'Images/{self.i_string}(c(2)).png')
        elif self.model_name == 'baseline_most_frequent':
            plt.savefig(f'Images/{self.i_string}(c(3)).png')
        elif self.model_name == 'baseline_random':
            plt.savefig(f'Images/{self.i_string}(c(4)).png')
        plt.close()
        
        return cm

    def plot_decision_boundary(self, index):
        xx, yy = np.meshgrid(np.linspace(self.X_train[:, 0].min() - 1, self.X_train[:, 0].max() + 1, 100),
                             np.linspace(self.X_train[:, 1].min() - 1, self.X_train[:, 1].max() + 1, 100))
        X_poly_mesh = self.poly.transform(np.c_[xx.ravel(), yy.ravel()])
        Z = self.final_model.predict(X_poly_mesh)
        Z = Z.reshape(xx.shape)

        plt.figure(figsize=(10, 6))
        plt.contourf(xx, yy, Z, alpha=0.3)
        plt.scatter(self.X_train[:, 0], self.X_train[:, 1], c=self.y_train, edgecolors='k', marker='o')
        plt.title(f'Decision boundary of {self.model_name} (week4_{index}.csv)')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        if self.model_name == 'log_reg':
            plt.savefig(f'Images/{self.i_string}(a(3)).png')
        elif self.model_name == 'knn':
            plt.savefig(f'Images/{self.i_string}(b(3)).png')
        plt.close()

    def plot_roc_curve(self, index):
        fpr, tpr, _ = roc_curve(self.y_test, self.y_prob)
        auc_score = auc(fpr, tpr)

        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, label=f'{self.model_name} (AUC = {auc_score:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve (week4_{index}.csv)')
        plt.legend(loc='best')
        if self.model_name == 'log_reg':
            plt.savefig(f'Images/{self.i_string}(d(1)).png')
        elif self.model_name == 'knn':
            plt.savefig(f'Images/{self.i_string}(d(2)).png')
        elif self.model_name == 'baseline_most_frequent':
            plt.savefig(f'Images/{self.i_string}(d(3)).png')
        elif self.model_name == 'baseline_random':
            plt.savefig(f'Images/{self.i_string}(d(4)).png')
        plt.close()
