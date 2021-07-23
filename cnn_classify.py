#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 14:43:44 2021

@author: hp
"""

from keras import layers
from keras import Input
from keras.layers import Dense, Dropout
from keras.models import Model
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MaxPooling2D
from keras.optimizers import SGD, Adam
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16

np.set_printoptions(threshold=np.inf)
x_train = np.load(f'/home/hp/.config/spyder-py3/music-speech/wavfile/train_data/X_train.npy')
y_train = np.load(f'/home/hp/.config/spyder-py3/music-speech/wavfile/train_data/Y_train.npy')
x_test = np.load(f'/home/hp/.config/spyder-py3/music-speech/wavfile/test_data/X_test.npy')
y_test = np.load(f'/home/hp/.config/spyder-py3/music-speech/wavfile/test_data/Y_test.npy')
'''
print(x_train.shape)
print(y_train.shape)

print(x_test.shape)
print(y_test.shape)

x_train = x_train/255
x_test = x_test/255
'''
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)

x_train, x_val, y_train_one_hot, y_val = train_test_split(x_train, y_train_one_hot, test_size=0.2)

                 
IMG_SIZE = (288,432)

IMG_SHAPE = IMG_SIZE + (3,)  
rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset= -1,input_shape = IMG_SHAPE)

base_model = VGG16(weights='imagenet',
                  include_top=False,              
                  input_shape=IMG_SHAPE)
base_model.trainable = False

model = tf.keras.Sequential()
model.add(rescale)
model.add(base_model)
model.add(tf.keras.layers.Conv2D(64, (3, 3),  strides=(1, 1), padding = 'same', activation = 'relu'))
model.add(tf.keras.layers.Conv2D(32, (3, 3),  strides=(1, 1), padding = 'same', activation = 'relu'))    #   161 161 32
model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding = 'valid'))    
model.add(tf.keras.layers.GlobalAveragePooling2D())    
model.add(tf.keras.layers.Dense(16, activation='relu'))

model.add(tf.keras.layers.Dense(3, activation='softmax')) 


model.compile(loss='categorical_crossentropy', optimizer='Adam',metrics=['accuracy'])

batch_size = 20
epochs = 10
history = model.fit(x_train, y_train_one_hot, epochs=epochs, batch_size = batch_size, validation_data=(x_val, y_val))

score = model.evaluate(x_test, y_test_one_hot, batch_size=20)

summary=model.summary()
print(summary)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

test_predictions = model.predict(x_test)
print(np.argmax(test_predictions[1]))

print("Baseline Error: %.2f%%" % (100-score[1]*100))

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Classifier Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train','Val'], loc = 'upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Classifier Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train','Val'], loc = 'upper left')
plt.show()

   
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns  
confusion = confusion_matrix(y_test, np.argmax(test_predictions,axis=1))
fig,ax=plot_confusion_matrix(conf_mat=confusion, figsize=(8,6))
plt.tight_layout
'''
import csv,os
header = 'filename Predictions Ground_Truth'
header = header.split()

filename = open('cnn_result.csv', 'w', newline='')
with filename:
    writer = csv.writer(filename)
    writer.writerow(header)
classes='speech music m+s'.split()
for c in classes:    
  for files in os.listdir(f'/home/hp/.config/spyder-py3/music-speech/wavfile/test/{c}'):
        file = f'/home/hp/.config/spyder-py3/music-speech/wavfile/test/{c}/{files}'
        to_append = f'{file}'
        to_append += f' {c}'
        file = open('cnn_result.csv', 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())
predicted_classes = np.argmax(np.round(test_predictions),axis=1)
#print(predicted_classes.shape)
#print(y_test.shape)
import pandas as pd
results=pd.DataFrame({"Predictions":predicted_classes, "Ground Truth":y_test})
results.to_csv("cnn_v",index=False)
'''
results=[]
labels=np.array([])
labels=y_test
predicted_classes = np.argmax(np.round(test_predictions),axis=1)
results=np.equal(labels, predicted_classes)

import csv,os

header = 'Filename Test_Predictions Test_Predictions Test_Predictions Predicted_Class Ground_truth'
header = header.split()

output = open('CNN_result.csv', 'w+', newline='') 
with output:
    writer = csv.writer(output)
    writer.writerow(header)
classes=['speech', 'music', 'm+s']
for c in classes:    
  for files in os.listdir(f'/home/hp/.config/spyder-py3/music-speech/wavfile/test/{c}'):
        file = f'/home/hp/.config/spyder-py3/music-speech/wavfile/test/{c}/{files}'
        to_append = f'{file}'
        #to_append += f' {c}'
        file = open('CNN_result.csv', 'a')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())
            
with open("CNN_result.csv","a" ,newline='') as my_csv:
    newarray = csv.writer(my_csv,delimiter=',')
    newarray.writerows(test_predictions)
    
import pandas as pd
result=pd.DataFrame({"class":predicted_classes, 
                      "labels":y_test, "Correction":results})
result.to_csv("CNN_result.csv",mode='a',index=False)