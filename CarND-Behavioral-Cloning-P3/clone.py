import os
import csv
import cv2
import numpy as np
import sklearn
from sklearn.utils import resample
from sklearn.model_selection import train_test_split

samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    # next(reader, None)  # skip the headers
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# define generator
def generator(samples, batch_size=32):
    while 1: # Loop forever so the generator never terminates
        images = []
        angles = []

        batch_samples = resample(samples, n_samples=batch_size)

        for batch_sample in batch_samples:
            angle_center = float(batch_sample[3])
            # create adjusted steering measurements for the side camera images
            # correction = 0.03 # this is a parameter to tune
            # angle_left = angle_center + correction
            # angle_right = angle_center - correction

            name_center = './data/IMG/'+batch_sample[0].split('\\')[-1]
            # name_left   = './data/IMG/'+batch_sample[1].split('\\')[-1]
            # name_right  = './data/IMG/'+batch_sample[2].split('\\')[-1]
            image_center = cv2.imread(name_center)
            # image_left   = cv2.imread(name_left)
            # image_right  = cv2.imread(name_right)

            # images.extend([image_center, image_left, image_right])
            # angles.extend([angle_center, angle_left, angle_right])
            images.extend([image_center])
            angles.extend([angle_center])            

            # Flip the image and steering angle
            image_center_flipped = np.fliplr(image_center)
            angle_center_flipped = -angle_center
            # image_left_flipped = np.fliplr(image_left)
            # angle_left_flipped = -angle_left
            # image_right_flipped = np.fliplr(image_right)
            # angle_right_flipped = -angle_right
            # images.extend([image_center_flipped, image_left_flipped, image_right_flipped])
            # angles.extend([angle_center_flipped, angle_left_flipped, angle_right_flipped])
            images.extend([image_center_flipped])
            angles.extend([angle_center_flipped])

        # trim image to only see section with road
        X_train = np.array(images)
        y_train = np.array(angles)
        yield X_train, y_train

# compile and train the model using the generator function
train_generator      = generator(train_samples, batch_size=128)
validation_generator = generator(validation_samples, batch_size=128)

from keras.models import Sequential
from keras.layers import Lambda, Input, Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

# Nvida End to End model
model = Sequential()
# set up cropping2D layer
model.add(Cropping2D(cropping=((60,25), (0,0)), input_shape=(160,320,3)))
# normalization and centering
model.add(Lambda(lambda x: (x / 255.0) - 0.5))
# add convolution layer and relu activation
model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation="relu"))
# model.add(Dropout(0.2))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation="relu"))
# model.add(Dropout(0.2))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation="relu"))
# model.add(Dropout(0.2))
model.add(Convolution2D(64, 3, 3, activation="relu"))
# model.add(Dropout(0.2))
model.add(Convolution2D(64, 3, 3, activation="relu"))
# model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(100))
# model.add(Dropout(0.2))
model.add(Dense(50))
# model.add(Dropout(0.2))
model.add(Dense(10))
# model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

history_object = model.fit_generator(train_generator, 
  									 samples_per_epoch= len(train_samples), 
									 validation_data=validation_generator, 
									 nb_val_samples=len(validation_samples), 
									 nb_epoch=10)

model.save('model_nvidia.h5')

### print the keys contained in the history object
print(history_object.history.keys())

import matplotlib.pyplot as plt
### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
plt.savefig('loss.jpg')