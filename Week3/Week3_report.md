<div align="center">

# Assignment Week 3
## Name: Patrick Farmer Student Number 201331828

### Dataset: 20--40-20 

</div>


### I(A)
The data was read in from the csv into X which held feature 1 and 2 and to Y which held the target. The data was then plotted onto a 3d scatter plot. We can see from the four different angles of the plot below that the data lies on a curve.\
![](i(a).png)\
The code to produce this plot is as follows:

<div style="font-size: 0.8em;">

```python
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
```
</div>

### I(B)
Features up to the power of 5 were added to the dataset by using the PolynomialFeatures function in sklearn. Using this new data, a lasso regression model was trained with a number of different c values. The C Values chosen were (1, 10, 1000 and 10000). The value 1 was started with because it had all coefficients of 0. The value 10 was then chosen second as it demonstrated a significant improvement in the accuracy of the model with a mean square error of 0.7, there was only 2 coefficients that were non 0 in this case, 
