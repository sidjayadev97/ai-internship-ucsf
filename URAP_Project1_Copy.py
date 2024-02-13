#!/usr/bin/env python
# coding: utf-8

# In[6]:


import matplotlib.pyplot as plt
import os
import numpy as np
os.getcwd()


# Alright, time to build a Neural Network!

# Before we start building the "ResNet architecture", let's start learning
# by buildinga the 'simple version' of a convolutional neural network (2D CNN)

# It works by taking the image data, using a filter on the image (called 'convolving')
# in order to detect edges, dark and bright spots, and other image characteristics
# Then these pieces of data are combined, pooled (meaning finding the max. value in
# a filter window), and then put through a dense connected layer.

# This system of Conv. -> Max Pooling -> Dense Layer is repeated 2-3 times,
# and finally we will have a Dropout layer (to correct for overfitting).

# First, we need to import Tensorflow and Keras
# Then import all the layers needed for the ConvNet 

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, InputLayer


# In[8]:


import os
project_folder = '/Users/rabeya/URAP/UCSF_Lab/Practice_Project_1/'
#os.listdir(project_folder+'Training_npy_images/')


# In[9]:



# Let's first work with our training set (and do a validation holdout set on it)
# Load the NumpyZ files
loaded_training_images_list = []
loaded_training_labels = []
training_file_folder = os.listdir(project_folder+'Training_npy_images/')
for file in training_file_folder:
    loaded = np.load(project_folder+'Training_npy_images/'+file)
    X = loaded['a']
    y_label = loaded['b']
    loaded_training_images_list.append(X)
    loaded_training_labels.append(y_label)


# In[10]:


import pandas as pd
labels_series = pd.Series([label.item(0) for label in loaded_training_labels])
#labels_series.head()


# In[11]:


# Now, we need to convert the y_labels into a numeric array
# We can code "DENSE" as 1, and "NOT_DENSE" as 0
label_mapping = {'DENSE':1, 'NOT_DENSE':0}
y_numeric = np.array(labels_series.map(label_mapping))


plt.imshow(loaded_training_images_list[0])


# Convert the iamge list and label list to numpy arrays
X = np.array(loaded_training_images_list)
y = y_numeric


#img0_copies = [X[0], X[0], X[0]]
#np.stack(img0_copies, axis=-1).shape



X_stacked = np.array([np.stack([img, img, img], axis=-1) for img in X])



# Let's make the training and validation subsets (from the training data itself!)
X_train, X_valid = X_stacked[:160], X_stacked[160:]
y_train, y_valid = y[:160], y[160:]




# Now that we have a numerical encoding of our breast density
# labels, let's start building the 2D ConvNet model!

from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Dense, Flatten
from keras.models import Sequential
from keras import optimizers

model = Sequential()

# First layer
model.add(Conv2D(32, (3, 3),  input_shape=(299, 299, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2)))


# Second layer
model.add( Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2)))

# Third layer
model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2)))

# Final Layer
model.add(Flatten())
model.add(Dropout(0.5))
          
          
# Dense Layer
model.add(Dense(64))
model.add(Activation('relu'))

# Output Layer
model.add(Dense(1))
model.add(Activation('sigmoid'))


# In[24]:


model.summary()


# Compile the CNN model
model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=['accuracy'])


from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img

# Code mimicked from "http://towardsdatascience.com"(Image-Detection-Keras article)
batch = 8
train_datagen = ImageDataGenerator()
val_datagen = ImageDataGenerator()

train_generator = train_datagen.flow(X_train, y_train, batch_size=batch)

val_generator = val_datagen.flow(X_valid, y_valid, batch_size=batch)


# Code mimicked from "http://towardsdatascience.com"(Image-Detection-Keras article)
n_train = len(X_train)
n_val = len(X_valid)
history = model.fit_generator(train_generator,
                             steps_per_epoch=n_train//batch,
                             epochs=64,
                             validation_data=val_generator,
                             validation_steps=n_val//batch)
