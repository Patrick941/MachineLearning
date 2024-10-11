from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import cross_val_score

file_path = os.path.join(os.path.dirname(__file__), 'week3.csv')
column_names = ['Feature1', 'Feature2', 'Target']
data = pd.read_csv(file_path, names=column_names, skiprows=1)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data['Feature1'], data['Feature2'], data['Target'], color='red')

ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Target')
ax.set_title('3D Scatter Plot of Base Data')

fig, axes = plt.subplots(2, 2, subplot_kw={'projection': '3d'}, figsize=(12, 12))

angles = [(20, 30), (30, 60), (40, 90), (10, -60)]
for ax, (elev, azim) in zip(axes.flatten(), angles):
    ax.scatter(data['Feature1'], data['Feature2'], data['Target'], color='red')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Target')
    ax.set_title(f'3D Scatter Plot (elev={elev}, azim={azim})')
    ax.view_init(elev=elev, azim=azim)

output_path = os.path.join(os.path.dirname(__file__), 'i(a).png')
plt.savefig(output_path)
plt.close()

def regression_analysis(model_class, C_values, png_name):
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

    for C in C_values:
        model = model_class(alpha=1/(2*C), max_iter=10000)
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
        model = model_class(alpha=1/(2*C), max_iter=10000)
        model.fit(X_poly, y)
        predictions = model.predict(X_test_poly)
        predictions = predictions.reshape((50, 50))
        
        ax.plot_trisurf(X_test[:, 0], X_test[:, 1], predictions.flatten(), color=color, alpha=0.5)
        ax.scatter(X[:, 0], X[:, 1], y, color='red', label='Training Data')
        
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_zlabel('Target')
        ax.set_title(f'{model_class.__name__} predictions C = {C}')
        
        ax.legend()

    output_path = os.path.join(os.path.dirname(__file__), png_name)
    plt.savefig(output_path)

print("\033[91m" + "Lasso regression:" + "\033[0m")
C_values = [1, 10, 1000, 10000]
regression_analysis(Lasso, C_values, 'i(c).png')

print("\033[91m" + "Ridge regression:" + "\033[0m")
C_values = [0.000001, 0.001, 1, 10]
regression_analysis(Ridge, C_values, 'i(e).png')

def cross_validation_analysis(model_class, C_values, png_name):
    file_path = os.path.join(os.path.dirname(__file__), 'week3.csv')
    column_names = ['Feature1', 'Feature2', 'Target']
    data = pd.read_csv(file_path, names=column_names, skiprows=1)

    X = data[['Feature1', 'Feature2']].values
    y = data['Target'].values

    poly = PolynomialFeatures(degree=5)
    X_poly = poly.fit_transform(X)

    mean_errors = []
    std_errors = []

    previous_mean_error = None
    threshold = 0.01 
    meaningful_data_found = False
    initial_c = None
    final_c = None
    spaced_c_values = []

    for C in C_values:
        model = model_class(alpha=1/(2*C), max_iter=10000)
        scores = cross_val_score(model, X_poly, y, cv=5, scoring='neg_mean_squared_error')
        mean_error = -scores.mean()
        std_error = scores.std()

        mean_errors.append(mean_error)
        std_errors.append(std_error)

        if previous_mean_error is not None:
            error_difference = abs(previous_mean_error - mean_error)
            if error_difference < threshold:
                if meaningful_data_found == True:
                    final_c = previous_c
                else:
                    pass
            else:
                if meaningful_data_found == True:
                    pass
                else: 
                    initial_c = previous_c              
                meaningful_data_found = True            
        else:
            pass

        previous_c = C
        previous_mean_error = mean_error
        
    mean_errors = []
    std_errors = []
        
    if initial_c is not None and final_c is not None:
        spaced_c_values = np.logspace(np.log10(initial_c), np.log10(final_c), 20)
        
    for spaced_C in spaced_c_values:
        model = model_class(alpha=1/(2*spaced_C), max_iter=10000)
        scores = cross_val_score(model, X_poly, y, cv=5, scoring='neg_mean_squared_error')
        mean_error = -scores.mean()
        std_error = scores.std()

        mean_errors.append(mean_error)
        std_errors.append(std_error)

    plt.figure()   
    plt.errorbar(spaced_c_values, mean_errors, yerr=std_errors, fmt='o-', capsize=5)
    plt.xscale('log')
    plt.xlabel('C values (log scale)')
    plt.ylabel('Mean squared error')
    plt.title(f'Cross-validation error for {model_class.__name__}')
    
    output_path = os.path.join(os.path.dirname(__file__), png_name)
    plt.savefig(output_path)
    plt.close()
print("\033[91m" + "Lasso regression:" + "\033[0m")
C_values = [10**i for i in range(-7, 5)]
cross_validation_analysis(Lasso, C_values, 'ii(a).png')
print("\033[91m" + "Ridge regression:" + "\033[0m")
cross_validation_analysis(Ridge, C_values, 'ii(c).png')
