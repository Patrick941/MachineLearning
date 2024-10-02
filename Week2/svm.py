import numpy as np
import pandas as pd
import os
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

fileDirectory = os.path.dirname(os.path.abspath(__file__))
## Assumes only one csv
csv_file = [f for f in os.listdir(fileDirectory) if f.endswith('.csv')][0]
csv_path = os.path.join(fileDirectory, csv_file)
df = pd.read_csv(csv_path, skiprows=1)
X1 = df.iloc[:, 0]
X2 = df.iloc[:, 1]
X = np.column_stack((X1, X2))
y = df.iloc[:, 2]

plt.figure(figsize=(8, 6))
plt.scatter(X[y == 1, 0], X[y == 1, 1], marker='o', color='blue', label='+1')
plt.scatter(X[y == -1, 0], X[y == -1, 1], marker='o', color='red', label='-1')

plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend()
plt.title('2D Plot of Features with Target Values')
plt.grid(True)
plt.savefig(os.path.join(fileDirectory, 'A(i).png'))

C_values = [0.01, 1, 100, 1000]
models = []
for C in C_values:
    model = make_pipeline(StandardScaler(), SVC(kernel='rbf', degree=2, C=C, max_iter=1000000))
    model.fit(X, y)
    models.append((C, model))

for C, model in models:
    print(f"Model trained with C={C}")
    print(f"Support Vectors: {model.named_steps['svc'].support_vectors_}")

x_min, x_max = -1, 1
y_min, y_max = -1, 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                        np.arange(y_min, y_max, 0.01))

fig, axes = plt.subplots(1, len(C_values), figsize=(20, 5))

for ax, (C, model) in zip(axes, models):
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.3)
    ax.scatter(X[y == 1, 0], X[y == 1, 1], marker='o', color='blue', label='Actual +1')
    ax.scatter(X[y == -1, 0], X[y == -1, 1], marker='o', color='red', label='Actual -1')
    
    predictions = model.predict(X)
    ax.scatter(X[predictions == 1, 0], X[predictions == 1, 1], marker='x', color='green', label='Predicted +1')
    ax.scatter(X[predictions == -1, 0], X[predictions == -1, 1], marker='+', color='orange', label='Predicted -1')
    
    ax.set_title(f'Decision Boundary for C={C}')
    ax.set_xlabel('x_1')
    ax.set_ylabel('x_2')
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(fileDirectory, 'B(ii).png'))
