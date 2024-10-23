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

def run_models():
	num_classes = 10
	input_shape = (32, 32, 3)
	(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
	x_train = x_train.astype("float32") / 255
	x_test = x_test.astype("float32") / 255
	print("orig x_train shape:", x_train.shape)

	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_test = keras.utils.to_categorical(y_test, num_classes)
	use_saved_model = False

	best_model = None
	best_model_description = ""
	best_results = model_runner.Results(0, 0, 0, 0)

	class_counts = np.bincount(np.argmax(y_train, axis=1))
	most_common_class = np.argmax(class_counts)
	y_pred_baseline = np.full(y_test.shape[0], most_common_class)
	y_pred_baseline = keras.utils.to_categorical(y_pred_baseline, num_classes)

	baseline_accuracy = np.mean(np.argmax(y_test, axis=1) == np.argmax(y_pred_baseline, axis=1)) * 100
	print("\033[93mBaseline Classifier Accuracy:\033[0m {:.2f}%".format(baseline_accuracy))

	training_data_sizes = [5000, 10000, 20000, 40000]

	for training_data_size in training_data_sizes:
		if training_data_size == 5000:
			regularisation_sizes = [0.0001, 0.001, 0.01, 0.1, 0, 1]
		else:
			regularisation_sizes = [0.0001]
		for regularisation_size in regularisation_sizes:
			if use_saved_model:
				model = keras.models.load_model("cifar.keras")
			else:
				x_train_subset = x_train[:training_data_size]
				y_train_subset = y_train[:training_data_size]
	
				model_run = model_runner.ModelRunner(x_train_subset, y_train_subset, x_test, y_test, num_classes, regularisation_size, "strides", 20)
				model_run.train_and_evaluate(training_data_size)
				model_results = model_run.results
				print(f"\033[93mStrides Model Results for {training_data_size}_{regularisation_size}:\033[0m", model_results)
				if (model_results.test_accuracy > best_results.test_accuracy):
					best_results = model_results
					best_model = model_run.model
					best_model_description = "Strides_" + str(training_data_size) + "_" + str(regularisation_size)
				if (training_data_size == 5000 and regularisation_size == 0.0001):
					model_run.model.summary()
				
				model_run = model_runner.ModelRunner(x_train_subset, y_train_subset, x_test, y_test, num_classes, regularisation_size, "pooling", 20)
				model_run.train_and_evaluate(training_data_size)
				model_results = model_run.results
				print(f"\033[93mPooling Model Results for {training_data_size}_{regularisation_size}:\033[0m", model_results)
				if (model_results.test_accuracy > best_results.test_accuracy):
					best_results = model_results
					best_model = model_run.model
					best_model_description = "Pooling_" + str(training_data_size) + "_" + str(regularisation_size)
				
				model_run = model_runner.ModelRunner(x_train_subset, y_train_subset, x_test, y_test, num_classes, regularisation_size, "advanced", 70)
				model_run.build_advanced_model()
				model_run.train_and_evaluate(training_data_size)
				model_results = model_run.results
				print(f"\033[93mAdvanced Model Results for {training_data_size}_{regularisation_size}:\033[0m", model_results)
				if (model_results.test_accuracy > best_results.test_accuracy):
					best_results = model_results
					best_model = model_run.model
					best_model_description = "Advanced_" + str(training_data_size) + "_" + str(regularisation_size)
		
	print("\033[93mBest Model Description:\033[0m", best_model_description)
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
