import numpy as np
import pandas as pd
import os

fileDirectory = os.path.dirname(os.path.abspath(__file__))
## Assumes only one csv
csv_file = [f for f in os.listdir(fileDirectory) if f.endswith('.csv')][0]
csv_path = os.path.join(fileDirectory, csv_file)
df = pd.read_csv(csv_path)

X1=df.iloc[:,0]
X2=df.iloc[:,1]
X=np.column_stack((X1,X2))
y=df.iloc[:,2]