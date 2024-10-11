from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

def load_data(file_name):
    file_path = os.path.join(os.path.dirname(__file__), file_name)
    column_names = ['Feature1', 'Feature2', 'Target']
    return pd.read_csv(file_path, names=column_names, skiprows=1)

def plot_3d_scatter(data, angles, output_file):
    fig, axes = plt.subplots(2, 2, subplot_kw={'projection': '3d'}, figsize=(12, 12))
    for ax, (elev, azim) in zip(axes.flatten(), angles):
        ax.scatter(data['Feature1'], data['Feature2'], data['Target'], color='red')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_zlabel('Target')
        ax.set_title(f'3D Scatter Plot')
        ax.view_init(elev=elev, azim=azim)
    plt.savefig(output_file)
    plt.close()

def regression_analysis(model_class, C_values, png_name):
    data = load_data('week3.csv')
    X = data[['Feature1', 'Feature2']].values
    y = data['Target'].values

    poly = PolynomialFeatures(degree=5)
    X_poly = poly.fit_transform(X)

    for C in C_values:
        model = model_class(alpha=1/(2*C), max_iter=10000)
        model.fit(X_poly, y)
        
        print(f'C: {C}')
        print(f'Coefficients: {model.coef_}')
        print(f'Intercept: {model.intercept_}')
        error = mean_squared_error(y, model.predict(X_poly))
        print(f'Mean square error: {error}')
        print("\n")

    plot_regression_results(model_class, C_values, X, y, X_poly, poly, png_name)

def plot_regression_results(model_class, C_values, X, y, X_poly, poly, png_name):
    X_test = []
    for i in np.linspace(-5, 5):
        for j in np.linspace(-5, 5):
            X_test.append([i, j])
    X_test = np.array(X_test)
    X_test_poly = poly.transform(X_test)

    fig, axes = plt.subplots(1, 4, subplot_kw={'projection': '3d'}, figsize=(24, 6))
    colors = ['blue', 'green', 'orange', 'purple']

    fig, axes = plt.subplots(2, 2, subplot_kw={'projection': '3d'}, figsize=(12, 12))
    colors = ['blue', 'green', 'orange', 'purple']

    for ax, C, color in zip(axes.flatten(), C_values, colors):
        model = model_class(alpha=1/(2*C), max_iter=10000)
        model.fit(X_poly, y)
        predictions = model.predict(X_test_poly).reshape((50, 50))
        
        ax.plot_surface(X_test[:, 0].reshape((50, 50)), X_test[:, 1].reshape((50, 50)), predictions, color=color, alpha=0.5)
        ax.scatter(X[:, 0], X[:, 1], y, color='red', label='Training Data')
        
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_zlabel('Target')
        ax.set_title(f'{model_class.__name__} predictions C = {C}')
        ax.legend()

    plt.savefig(os.path.join(os.path.dirname(__file__), png_name))
    plt.close()

def cross_validation_analysis(model_class, C_values, png_name):
    data = load_data('week3.csv')
    X = data[['Feature1', 'Feature2']].values
    y = data['Target'].values

    poly = PolynomialFeatures(degree=5)
    X_poly = poly.fit_transform(X)

    initial_c, final_c = find_meaningful_c_range(model_class, C_values, X_poly, y)
    if initial_c and final_c:
        spaced_c_values = np.logspace(np.log10(initial_c), np.log10(final_c), 20)
        plot_cross_validation_error(model_class, spaced_c_values, X_poly, y, png_name)

def find_meaningful_c_range(model_class, C_values, X_poly, y):
    previous_mean_error = None
    threshold = 0.01 
    meaningful_data_found = False
    initial_c = None
    final_c = None

    for C in C_values:
        model = model_class(alpha=1/(2*C), max_iter=10000)
        scores = cross_val_score(model, X_poly, y, cv=5, scoring='neg_mean_squared_error')
        mean_error = -scores.mean()

        if previous_mean_error is not None:
            error_difference = abs(previous_mean_error - mean_error)
            if error_difference < threshold:
                if meaningful_data_found:
                    final_c = previous_c
                    break
            else:
                if meaningful_data_found:
                    pass
                else: 
                    initial_c = previous_c              
                meaningful_data_found = True            
        previous_c = C
        previous_mean_error = mean_error

    return initial_c, final_c

def plot_cross_validation_error(model_class, C_values, X_poly, y, png_name):
    mean_errors = []
    std_errors = []

    for C in C_values:
        model = model_class(alpha=1/(2*C), max_iter=10000)
        scores = cross_val_score(model, X_poly, y, cv=5, scoring='neg_mean_squared_error')
        mean_errors.append(-scores.mean())
        std_errors.append(scores.std())

    plt.figure()   
    plt.errorbar(C_values, mean_errors, yerr=std_errors, fmt='o-', capsize=5)
    plt.xscale('log')
    plt.xlabel('C values (log scale)')
    plt.ylabel('Mean squared error')
    plt.title(f'Cross-validation error for {model_class.__name__}')
    
    plt.savefig(os.path.join(os.path.dirname(__file__), png_name))
    plt.close()

data = load_data('week3.csv')
angles = [(20, 30), (30, 60), (40, 90), (10, -60)]
plot_3d_scatter(data, angles, 'i(a).png')

print("\033[91m" + "Lasso regression:" + "\033[0m")
C_values = [1, 10, 1000, 10000]
regression_analysis(Lasso, C_values, 'i(c).png')

print("\033[91m" + "Ridge regression:" + "\033[0m")
C_values = [0.000001, 0.001, 1, 10]
regression_analysis(Ridge, C_values, 'i(e).png')

C_values = [10**i for i in range(-7, 5)]
cross_validation_analysis(Lasso, C_values, 'ii(a).png')
cross_validation_analysis(Ridge, C_values, 'ii(c).png')