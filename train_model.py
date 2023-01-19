# coding: utf-8

# Building model for Self-Driving Car
# Project 3 - Term 1, Self-Driving Car Nanodegree Program by Udacity

import numpy as np
import cv2
from keras.applications.vgg16 import VGG16
from keras.models import Model
import matplotlib.pyplot as plt
from keras.layers import Cropping2D, Flatten, Dense, Lambda, Dropout
from keras import __version__ as keras_version

print(keras_version)

# Load images and streering data
# The data set in this project is provided by Udacity, collected from Self-Driving Car simulator

lines = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []

for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = 'data/IMG/' + filename
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

# Data Augmentation: Flipping Images and Steering Measurements
# This is an effective technique for helping with the left turn bias involves flipping images 
# and taking the opposite sign of the steering measurements

augmented_images = []
augmented_measurements = []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_images.append(cv2.flip(image, 1))
    augmented_measurements.append(measurement)
    augmented_measurements.append(measurement * - 1.0)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

# Load the VGG16 model, pre-trained on ImageNet
model = VGG16(weights='imagenet', include_top=True)

# Preprocessing the data
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))  # Normalizing data + Mean centered data
model.add(Cropping2D(cropping=((70, 20), (0, 0))))  # Cropping image in Keras

# Remove the output layers of the VGG16 model
model.layers.pop()
model.layers.pop()
model.layers.pop()

# Add a new output layer for regression
model.add(Dense(1))

model.compile(loss = 'categorical_crossentropy', optimizer='adam')

history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=2)
model.save('model_10eps.h5')

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
