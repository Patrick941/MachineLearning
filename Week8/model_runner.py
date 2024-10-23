import time
import numpy as np
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dropout, Flatten, Dense, MaxPooling2D, BatchNormalization
from typing import NamedTuple
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam


import matplotlib.pyplot as plt

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
        
    def build_custom_model(self):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.4))
        model.add(Flatten())
        model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01)))
        model.add(Dropout(0.4))
        model.add(Dense(self.num_classes, activation='softmax'))
        optimizer = Adam(learning_rate=0.0001)
        model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
        
        self.model = model

    def train_and_evaluate(self, training_data_size):
        start_time = time.time()
        history = self.model.fit(self.x_train, self.y_train, batch_size=128, epochs=self.epochs, validation_split=0.1, verbose=0)
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