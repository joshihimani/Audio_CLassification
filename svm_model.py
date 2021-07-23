#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 17:29:53 2021

@author: hp
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt


data = pd.read_csv('dataset.csv')
data.head()

class_list = data.iloc[:, -1]
encoder = LabelEncoder()
y_train = encoder.fit_transform(class_list)

x_train = data.drop('label', axis=1)
print(x_train.shape)
print(y_train.shape)


data = pd.read_csv('dataset1.csv')
data.head()

class_list = data.iloc[:, -1]
encoder = LabelEncoder()
y_test = encoder.fit_transform(class_list)

x_test = data.drop('label', axis=1)

print(x_test.shape)
print(y_test.shape)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)

epochs=20
batch_size=20
svclassifier = SVC(kernel='rbf')

svclassifier.fit(x_train, y_train)

test_predictions = svclassifier.predict(x_test)

from sklearn.metrics import accuracy_score
print("Accuracy:", accuracy_score(y_test,test_predictions))

#predicted_classes = test_predictions
#print(predicted_classes)
#print(y_test.shape)

results=[]
labels=np.array([])
labels=y_test
results=np.equal(labels, test_predictions)

import csv,os

header = 'Filename Predicted_Class Ground_truth'
header = header.split()

output = open('SVM_result.csv', 'w+', newline='') 
with output:
    writer = csv.writer(output)
    writer.writerow(header)
classes=['speech', 'music', 'm+s']
for c in classes:    
  for files in os.listdir(f'/home/hp/.config/spyder-py3/music-speech/wavfile/test/{c}'):
        file = f'/home/hp/.config/spyder-py3/music-speech/wavfile/test/{c}/{files}'
        to_append = f'{file}'
        #to_append += f' {c}'
        file = open('SVM_result.csv', 'a')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())

    
import pandas as pd
result=pd.DataFrame({"class":test_predictions, 
                      "labels":y_test, "Correction":results})
result.to_csv("SVM_result.csv",mode='a',index=False)

print(test_predictions)
print(y_test)
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
confusion = confusion_matrix(y_test, test_predictions)
fig,ax=plot_confusion_matrix(conf_mat=confusion, figsize=(8,6))
plt.tight_layout

'''
import csv 
import os

header = 'filename Predictions Ground_Truth'
header = header.split()

filename = open('svm_r.csv', 'w', newline='')
with filename:
    writer = csv.writer(filename)
    writer.writerow(header)
classes='speech music m+s'.split()
for c in classes:    
  for files in os.listdir(f'/home/hp/.config/spyder-py3/music-speech/wavfile/test/{c}'):
        file = f'/home/hp/.config/spyder-py3/music-speech/wavfile/test/{c}/{files}'
        to_append = f'{file}'
        to_append += f' {c}'
        file = open('svm_r.csv', 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())


#print(predicted_classes.shape)
#print(y_test.shape)
import pandas as pd
results=pd.DataFrame({"Predictions":predicted_classes, "Ground Truth":y_test})
results.to_csv("pred_results.csv",index=False)
'''
