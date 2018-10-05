# -*- coding: utf-8 -*-

# Imports 
import numpy as np
import pandas as pd
import cv2

from os import path

# Loading the dataset
dataset = pd.read_csv(r'..\data\legend.csv')
dataset.columns = ['temp','file_name','label'] # Renaming the columns
dataset = dataset.drop('temp',axis = 1) # Dropping unessary axis

# Classes of dataset
classes = list(set(dataset['label']))

# Loading all Images and Labels
from keras.utils import normalize
images_dir = path.abspath(r'..\images') # Images directory
images_array = [] # Empty images array 
labels_array = [] # Empty labels array
for index, row in dataset.iterrows():
    labels_array.append(row['label'])
    #print(path.join(images_dir,row['file_name']),cv2.IMREAD_COLOR)
    image = cv2.imread(path.join(images_dir,row['file_name']),cv2.IMREAD_COLOR)
    #print(np.shape(image))
    resized = cv2.resize(image, (32,32), interpolation = cv2.INTER_CUBIC )
    resized = normalize(resized,axis =1 )
    images_array.append(resized)    
images_array = np.array(images_array) # Converting list into numpy array

# One Hot encoding
labels_array_numerical = [0]*len(labels_array)
for i in range(0,len(labels_array)):
    index = classes.index(labels_array[i])
    labels_array_numerical[i]= index
    
from keras.utils import to_categorical
labels_array = to_categorical(labels_array_numerical,num_classes =len(classes))

# Test Train Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(images_array, labels_array , test_size=0.25)

# Variables
epochs = 20

from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense
from keras import optimizers

# Building a model
model = Sequential()
model.add(Conv2D(64,input_shape =(32,32,3),kernel_size=(3,3),strides=(1,1), activation='relu',padding='valid',data_format="channels_last"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding='valid'))
model.add(Conv2D(32,kernel_size=(3,3),strides=(1,1), activation='relu',padding='valid'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding='valid'))
model.add(Conv2D(16,kernel_size=(3,3),strides=(1,1), activation='relu',padding='valid'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding='valid'))
model.add(Flatten(data_format="channels_last"))
model.add(Dense(512,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(len(classes),activation='softmax'))
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True) #Optimizer
model.compile(sgd, loss='categorical_crossentropy', metrics=['accuracy']) # Compiling the model

# Training the model    
model.fit(x=X_train, y=y_train, batch_size=32, epochs=epochs, verbose=1, callbacks=None, validation_data=(X_test,y_test), shuffle=True, steps_per_epoch=None)
