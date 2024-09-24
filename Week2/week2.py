import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

fileDirectory = os.path.dirname(os.path.abspath(__file__))
## Assumes only one csv
csv_file = [f for f in os.listdir(fileDirectory) if f.endswith('.csv')][0]
csv_path = os.path.join(fileDirectory, csv_file)
df = pd.read_csv(csv_path)

X1=df.iloc[:,0]
X2=df.iloc[:,1]
X=np.column_stack((X1,X2))
y=df.iloc[:,2]

plt.figure(figsize=(8, 6))
plt.scatter(X[y == 1, 0], X[y == 1, 1], marker='+', color='blue', label='+1')
plt.scatter(X[y == -1, 0], X[y == -1, 1], marker='o', color='red', label='-1')

plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend()
plt.title('2D Plot of Features with Target Values')
plt.grid(True)
plt.show()