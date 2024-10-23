import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # Log all messages
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, LeakyReLU
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import model_runner
plt.rc('font', size=18)
plt.rcParams['figure.constrained_layout.use'] = True
import sys


# Model / data parameters
num_classes = 10
input_shape = (32, 32, 3)

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
training_data_size=5000
x_train = x_train[1:training_data_size]; y_train=y_train[1:training_data_size]
#x_test=x_test[1:500]; y_test=y_test[1:500]

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
print("orig x_train shape:", x_train.shape)

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

training_data_sizes = [5000, 10000, 20000, 40000]
use_saved_model = False

best_model = None
best_results = model_runner.Results(0, 0, 0, 0)

for training_data_size in training_data_sizes:
	if training_data_size == 5000:
		regularisation_sizes = [0.0001, 0.001, 0.01, 0.1, 0, 1]
	else:
		regularisation_sizes = [0.001]
	for regularisation_size in regularisation_sizes:
		if use_saved_model:
			model = keras.models.load_model("cifar.keras")
		else:
			x_train_subset = x_train[:training_data_size]
			y_train_subset = y_train[:training_data_size]
   
			model_run = model_runner.ModelRunner(x_train_subset, y_train_subset, x_test, y_test, num_classes, regularisation_size, "strides")
			model_run.train_and_evaluate(training_data_size)
			model_results = model_run.results
			print("\033[93mStrides Model Results:\033[0m", model_results)
			if (model_results.validation_accuracy > best_results.validation_accuracy):
				best_results = model_results
				best_model = model_run.model
			
			model_run = model_runner.ModelRunner(x_train_subset, y_train_subset, x_test, y_test, num_classes, regularisation_size, "pooling")
			model_run.train_and_evaluate(training_data_size)
			model_results = model_run.results
			print("\033[93mPooling Model Results:\033[0m", model_results)
			if (model_results.validation_accuracy > best_results.validation_accuracy):
				best_results = model_results
				best_model = model_run.model
   			
			model_run = model_runner.ModelRunner(x_train_subset, y_train_subset, x_test, y_test, num_classes, regularisation_size, "strides")
			model_run.build_advanced_model()
			model_run.train_and_evaluate(training_data_size)
			model_results = model_run.results
			print("\033[93mAdvanced Model Results:\033[0m", model_results)
			if (model_results.validation_accuracy > best_results.validation_accuracy):
				best_results = model_results
				best_model = model_run.model
    
print("\033[93mBest Model Results:\033[0m", best_results)
   


# preds = model.predict(x_train)
# y_pred = np.argmax(preds, axis=1)
# y_train1 = np.argmax(y_train, axis=1)
# print(classification_report(y_train1, y_pred))
# print(confusion_matrix(y_train1,y_pred))
# 
# preds = model.predict(x_test)
# y_pred = np.argmax(preds, axis=1)
# y_test1 = np.argmax(y_test, axis=1)
# print(classification_report(y_test1, y_pred))
# print(confusion_matrix(y_test1,y_pred))