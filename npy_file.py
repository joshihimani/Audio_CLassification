#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 16:55:13 2021

@author: hp
"""


import os
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16
import cv2

conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))

x_train = []
y_train = []

classes=['speech', 'music', 'm+s']
input_path = f'/home/hp/.config/spyder-py3/music-speech/wavfile/data/' 
for clas in classes:
    files = os.listdir(input_path + clas)
    for file in files:
        filePath = input_path + clas + '/' + file
        img = cv2.imread(filePath)
        img = cv2.resize(img, (150, 150))
        img = np.expand_dims(img, axis=0)
        x_train.append(np.squeeze(conv_base.predict(img)))
        y_train.append(classes.index(clas))
        
x_train = np.asarray(x_train)
y_train = np.asarray(y_train)
print(x_train.shape)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1]*x_train.shape[2]*x_train.shape[3]))
np.save(f'/home/hp/.config/spyder-py3/music-speech/wavfile/data/x_train.npy', x_train)
np.save(f'/home/hp/.config/spyder-py3/music-speech/wavfile/data/y_train.npy', y_train)

input_path = f'/home/hp/.config/spyder-py3/music-speech/wavfile/tdata/' 

x_test = []
y_test = []

for clas in classes:
    files = os.listdir(input_path + clas)
    for file in files:
        filePath = input_path + clas + '/' + file
        img = cv2.imread(filePath)
        img = cv2.resize(img, (150, 150))
        img = np.expand_dims(img, axis=0)
        x_test.append(np.squeeze(conv_base.predict(img)))
        y_test.append(classes.index(clas))

x_test = np.asarray(x_test)
y_test = np.asarray(y_test)
print(x_test.shape)

x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1]*x_test.shape[2]*x_test.shape[3]))
np.save(f'/home/hp/.config/spyder-py3/music-speech/wavfile/tdata/x_test.npy', x_test)
np.save(f'/home/hp/.config/spyder-py3/music-speech/wavfile/tdata/y_test.npy', y_test) 








