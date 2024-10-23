---
geometry: margin=30mm
title: "Assignment Week 8"
author: "Student Name: Patrick Farmer       Student Number: 20501828"
date: "Date: 23-10-2024"
---

![](https://www.tcd.ie/media/tcd/site-assets/images/tcd-logo.png)

\clearpage

## I(a)
I began the assignment by implementing a function that convolves a an input array with a kernel and returns the result. The kernel is setup to be a argument to the function. The kernel is setup to iterate through the input array (image) from left to right and top to bottom all the while applying the kernel and saving the output to another array which is eventually returned. The function also does not perform any sort of padding which means that the output array will be smaller than the input array. If the same image is convolved a number of times there may be significant loss of information.

## I(b)
This section will implement the above function on an image. The image which I selected was the first I found being my desktop background. The Image is first resized to 200x200 and then loaded as three RGB arrays then only the red channel taken. The two kernels implemented are shown below:
$$
\begin{bmatrix}
-1 & -1 & -1 \\
-1 & 8 & -1 \\
-1 & -1 & -1
\end{bmatrix}
$$
$$
\begin{bmatrix}
0 & -1 & 0 \\
-1 & 8 & -1 \\
0 & -1 & 0
\end{bmatrix}
$$
The input image, output image for kernel 1 and output image for kernel 2 are shown below respectively:
![alt text](Images/original.png){ width=50% }
![alt text](Images/result1.png){ width=50% }
![alt text](Images/result2.png){ width=50% }\
The first image appears to show an edge image meaning that the kernel is an edge filter.\
The second image appears to show a sharpened image as can be seen by the increase in contrast between edges but also the overemphasis of noise. This means that the kernel is a sharpening filter.

## II(a)
The convolutional network architecture consists of four convolution layers with ReLU activation. Where the first two layers have 16 filters each while the other two have 32 filters each. The kernel size does not change between layers all of them have a 3x3 kernel size, with padding being used which means that after convolutional the original image size is recovered by mirroring the outermost neighbour pixel. The second and fourth layers implement a stride of 2x2 which represents how far across the image the kernel is moved before it is applied.
The regularisation that is used by this model is first a dropout layer of 0.5, this means that for each epoch half of the neurons will be shut off so that the model does not just learn a couple 'paths' to make it's decision. The second form of regularisation used is L1 regularisation with a value of 0.0001 which adds a penalty to the loss function for large weights. This is used to prevent overfitting.\
The model is then flattened into a 2d array. Finally a dense layer is applied with a softmax activation function. Each neuron is connected to every neuron in the previous layer. The amount of neurons matches the amount of output classes which in this case is 10. The softmax function is used to convert the output of the model into a probability distribution.\
The batch size is 128 and the model is trained for 20 epochs.

## II(b)(i)
The model when run with 5000 data points does get an accuracy close to 48% on the test data. In this can 51%. The training data performs much better with 63%. The model is overfitting but there is also few improvements to be made by increasing regularisation, the data set would need to be larger for more substantial improvements.\
The amount of parameters in this mode is 37,146 parameters and then 74,294 optimiser parameters. The layer with the largest parameter count is the dense layer with 20,490 parameters.\
When comparing against a baseline classifier that simply predicts the most common class the model is performing much better. The baseline classifier has an accuracy of 10% which means that the dataset is balanced amongst the classes otherwise the accuracy of baseline would be higher than 100%/10.

## II(b)(ii)
The validation accuracy and training accuracy of the model can be plotted over its epochs along with the loss of the model. This is shown in the below graph:\
![](Images/cifar_5000_0.0001_strides.png)\
As already stated by the difference in the training and validation accuracy we can see that this model is overfitting. From the graph we can tell this because the line for the training accuracy and validation accuracy begin to diverge from one another before the end of the training. We can also see it from the loss graph because the training loss continues to decrease while the validation loss becomes stagnant.

## II(b)(iii)

## II(b)(iv)

## II(c)(i)

## II(c)(ii)

## II(d)

## Appendices

### I(a)
```python
import numpy as np

def convolve(input_array, kernel):
    n = input_array.shape[0]
    k = kernel.shape[0]
    output_size = n - k + 1
    output_array = np.zeros((output_size, output_size))

    for i in range(output_size):
        for j in range(output_size):
            sub_array = input_array[i:i+k, j:j+k]
            product = sub_array * kernel
            sum_product = np.sum(product)
            output_array[i, j] = sum_product
    
    return output_array

```

### I(b)
```python
import convolution
from PIL import Image
import numpy as np
import os
import week8

downloads_folder = os.path.expanduser('~/Downloads')
image_files = [f for f in os.listdir(downloads_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

if not image_files:
    raise FileNotFoundError("No image files found in the downloads folder.")

first_image_path = os.path.join(downloads_folder, image_files[0])
im = Image.open(first_image_path)

width, height = im.size
new_height = 200
new_width = int((new_height / height) * width)
im = im.resize((new_width, new_height), Image.LANCZOS)
rgb = np.array(im.convert('RGB'))

images_folder = os.path.expanduser('Images')
os.makedirs(images_folder, exist_ok=True)
resized_image_path = os.path.join(images_folder, 'original.png')
im.save(resized_image_path)

r = rgb[:, :, 0]

kernel1 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
kernel2 = np.array([[0, -1, 0], [-1, 8, -1], [0, -1, 0]])

result1 = convolution.convolve(r, kernel1)
result2 = convolution.convolve(r, kernel2)

result1_image_path = os.path.join(images_folder, 'result1.png')
result2_image_path = os.path.join(images_folder, 'result2.png')

Image.fromarray(np.uint8(result1)).save(result1_image_path)
Image.fromarray(np.uint8(result2)).save(result2_image_path)

week8.run_models()
```

### II(a)
```python
model = keras.Sequential()
model.add(Conv2D(16, (3,3), padding='same', input_shape=x_train.shape[1:],activation='relu'))
model.add(Conv2D(16, (3,3), strides=(2,2), padding='same', activation='relu'))
model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
model.add(Conv2D(32, (3,3), strides=(2,2), padding='same', activation='relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(num_classes, activation='softmax',kernel_regularizer=regularizers.l1(0.0001)))
model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
model.summary()

batch_size = 128
epochs = 20
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
```
### II(b)(i)
```python
class_counts = np.bincount(np.argmax(y_train, axis=1))
most_common_class = np.argmax(class_counts)
y_pred_baseline = np.full(y_test.shape[0], most_common_class)
y_pred_baseline = keras.utils.to_categorical(y_pred_baseline, num_classes)

baseline_accuracy = np.mean(np.argmax(y_test, axis=1) == np.argmax(y_pred_baseline, axis=1)) * 100
print("\033[93mBaseline Classifier Accuracy:\033[0m {:.2f}%".format(baseline_accuracy))
```

### II(b)(iii)
```python
best_model = None
best_model_description = ""
best_results = model_runner.Results(0, 0, 0, 0)


training_data_sizes = [5000, 10000, 20000, 40000]

for training_data_size in training_data_sizes:
    model_run = model_runner.ModelRunner(x_train_subset, y_train_subset, x_test, y_test, num_classes, regularisation_size, "strides", 20)
    model_run.train_and_evaluate(training_data_size)
    model_results = model_run.results
    print("\033[93mStrides Model Results:\033[0m", model_results)
    if (model_results.validation_accuracy > best_results.validation_accuracy):
        best_results = model_results
        best_model = model_run.model
        best_model_description = "Strides_" + str(training_data_size) + "_" + str(regularisation_size)

class Results(NamedTuple):
    training_time: float
    train_accuracy: float
    test_accuracy: float
    validation_accuracy: float

class ModelRunner:
    def __init__(self, x_train, y_train, x_test, y_test, num_classes, regularisation_size, downsampling, epochs):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.num_classes = num_classes
        self.regularisation_size = float(regularisation_size)
        self.downsampling = downsampling
        self.model = self.build_model()
        self.results = Results(training_time=0.0, train_accuracy=0.0, test_accuracy=0.0, validation_accuracy=0.0)
        self.epochs = epochs

    def build_model(self):
        model = Sequential()
        model.add(Conv2D(16, (3,3), padding='same', input_shape=self.x_train.shape[1:], activation='relu'))
        if self.downsampling == "strides":
            model.add(Conv2D(16, (3,3), strides=(2,2), padding='same', activation='relu'))
        else:
            model.add(Conv2D(16, (3,3), padding='same', activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
        if self.downsampling == "strides":
            model.add(Conv2D(32, (3,3), strides=(2,2), padding='same', activation='relu'))
        else:
            model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(self.num_classes, activation='softmax', kernel_regularizer=regularizers.l1(self.regularisation_size)))
        model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
        return model

    def train_and_evaluate(self, training_data_size):
        start_time = time.time()
        history = self.model.fit(self.x_train, self.y_train, batch_size=128, epochs=self.epochs, validation_split=0.1)
        end_time = time.time()

        training_time = end_time - start_time

        # Evaluate on training data
        train_preds = self.model.predict(self.x_train)
        y_train_pred = np.argmax(train_preds, axis=1)
        y_train_true = np.argmax(self.y_train, axis=1)
        train_accuracy = np.mean(y_train_pred == y_train_true)

        # Evaluate on test data
        test_preds = self.model.predict(self.x_test)
        y_test_pred = np.argmax(test_preds, axis=1)
        y_test_true = np.argmax(self.y_test, axis=1)
        test_accuracy = np.mean(y_test_pred == y_test_true)
        
        self.results = Results(training_time=training_time, train_accuracy=train_accuracy, test_accuracy=test_accuracy, validation_accuracy=history.history['val_accuracy'][-1])

        # Plotting accuracy and loss
        plt.figure(figsize=(12, 6))
        plt.subplot(211)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title(f'Model accuracy for {training_data_size} samples with regularisation {self.regularisation_size} using {self.downsampling}')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.subplot(212)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title(f'Model loss for {training_data_size} samples with regularisation {self.regularisation_size} using {self.downsampling}')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(f"Images/cifar_{training_data_size}_{self.regularisation_size}_{self.downsampling}.png")
        plt.close()
        
    def get_results(self):
        return self.results
```

### II(b)(iv)
```python
for training_data_size in training_data_sizes:
    if training_data_size == 5000:
        regularisation_sizes = [0.0001, 0.001, 0.01, 0.1, 0, 1]
    else:
        regularisation_sizes = [0.001]
    for regularisation_size in regularisation_sizes:
        # Same code
```

### II(c)(i)
```python
model_run = model_runner.ModelRunner(x_train_subset, y_train_subset, x_test, y_test, num_classes, regularisation_size, "pooling", 20)
model_run.train_and_evaluate(training_data_size)
model_results = model_run.results
print("\033[93mPooling Model Results:\033[0m", model_results)
if (model_results.validation_accuracy > best_results.validation_accuracy):
    best_results = model_results
    best_model = model_run.model
    best_model_description = "Pooling_" + str(training_data_size) + "_" + str(regularisation_size)

class ModelRunner:
    def build_model(self):
        model = Sequential()
        model.add(Conv2D(16, (3,3), padding='same', input_shape=self.x_train.shape[1:], activation='relu'))
        if self.downsampling == "strides":
            model.add(Conv2D(16, (3,3), strides=(2,2), padding='same', activation='relu'))
        else:
            model.add(Conv2D(16, (3,3), padding='same', activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
        if self.downsampling == "strides":
            model.add(Conv2D(32, (3,3), strides=(2,2), padding='same', activation='relu'))
        else:
            model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(self.num_classes, activation='softmax', kernel_regularizer=regularizers.l1(self.regularisation_size)))
        model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
        return model
```

### II(d)
```python
model_run = model_runner.ModelRunner(x_train_subset, y_train_subset, x_test, y_test, num_classes, regularisation_size, "advanced", 70)
model_run.build_advanced_model()
model_run.train_and_evaluate(training_data_size)
model_results = model_run.results
print("\033[93mAdvanced Model Results:\033[0m", model_results)
if (model_results.validation_accuracy > best_results.validation_accuracy):
    best_results = model_results
    best_model = model_run.model
    best_model_description = "Advanced_" + str(training_data_size) + "_" + str(regularisation_size)

def build_advanced_model(self):
    model = Sequential()
    model.add(Conv2D(8, (3, 3), padding='same', input_shape=self.x_train.shape[1:], activation='relu'))
    model.add(Conv2D(8, (3, 3), strides=(2, 2), padding='same', activation='relu'))
    model.add(Conv2D(16, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(16, (3, 3), strides=(2, 2), padding='same', activation='relu'))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(32, (3, 3), strides=(2, 2), padding='same', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(self.num_classes, activation='softmax', kernel_regularizer=regularizers.l1(self.regularisation_size)))
    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
    self.model = model
```