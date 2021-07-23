#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 12:49:27 2021

@author: hp
"""
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.inf)

def data():
    x_train = np.load(f'/home/hp/.config/spyder-py3/music-speech/wavfile/data/x_train.npy')
    y_train = np.load(f'/home/hp/.config/spyder-py3/music-speech/wavfile/data/y_train.npy')
    x_test = np.load(f'/home/hp/.config/spyder-py3/music-speech/wavfile/tdata/x_test.npy')
    y_test = np.load(f'/home/hp/.config/spyder-py3/music-speech/wavfile/tdata/y_test.npy')

    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)
    return x_train,x_test,y_train,y_test

def normalize(x_train,x_test,y_train,y_test):
    x_train = x_train/255
    x_test = x_test/255

    y_train_one_hot = to_categorical(y_train)
    y_test_one_hot = to_categorical(y_test)
    return x_train,x_test,y_train_one_hot,y_test_one_hot,y_test

def split(x_train,y_train_one_hot):
    x_train, x_val, y_train_one_hot, y_val = train_test_split(x_train, y_train_one_hot, test_size=0.2)
    return x_val,y_val

def fc_model(x_train,y_train_one_hot,x_val,y_val,batch_size,epochs):
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(32, input_dim=x_train.shape[1], activation='relu'))
    model.add(Dense(16,  activation='relu'))
    #model.add(Dense(32,  activation='relu'))
    model.add(Dense(3, activation='softmax'))  
    model.compile(loss='categorical_crossentropy', optimizer='Adam',metrics=['accuracy'])
    history = model.fit(x_train, y_train_one_hot, epochs=epochs, batch_size = batch_size, validation_data=(x_val, y_val))
    return model,history
def evaluation(batch_size):
    score = model.evaluate(x_test, y_test_one_hot, batch_size=batch_size)
    summary=model.summary()
    print(summary)
    loss=score[0]
    accuracy=score[1]
    print('Test loss:', loss)
    print('Test accuracy:', accuracy)

    test_predictions = model.predict(x_test)
    print(test_predictions)
    print("Baseline Error: %.2f%%" % (100-score[1]*100))
    #print(np.argmax(test_predictions[1]))
    return loss, accuracy,test_predictions

def plotting(y_test,test_predictions):
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
    
def results(y_test):
    predicted_classes = np.argmax(np.round(test_predictions),axis=1)
#print(predicted_classes)
#print(y_test.shape)
    results=[]
    labels=np.array([])
    labels=y_test
    results=np.equal(labels, predicted_classes)

    import csv,os

    header = 'Filename Test_Predictions Test_Predictions Test_Predictions Predicted_Class Ground_truth'
    header = header.split()

    output = open('FC_result.csv', 'w+', newline='') 
    with output:
        writer = csv.writer(output)
        writer.writerow(header)
    classes=['speech', 'music', 'm+s']
    for c in classes:    
       for files in os.listdir(f'/home/hp/.config/spyder-py3/music-speech/wavfile/test/{c}'):
        file = f'/home/hp/.config/spyder-py3/music-speech/wavfile/test/{c}/{files}'
        to_append = f'{file}'
        #to_append += f' {c}'
        file = open('FC_result.csv', 'a')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())
            
    with open("FC_result.csv","a" ,newline='') as my_csv:
      newarray = csv.writer(my_csv,delimiter=',')
      newarray.writerows(test_predictions)
    
    import pandas as pd
    result=pd.DataFrame({"class":predicted_classes, 
                      "labels":y_test, "Correction":results})
    result.to_csv("FC_result.csv",mode='a',index=False)
   
if __name__ == "__main__":  
    batch_size=2
    epochs=20
    x_train,x_test,y_train,y_test=data()
    x_train,x_test,y_train_one_hot,y_test_one_hot,y_test=normalize(x_train,x_test,y_train,y_test)
    x_val,y_val=split(x_train,y_train_one_hot)
    model,history=fc_model(x_train,y_train_one_hot,x_val,y_val,batch_size,epochs)
    loss, accuracy,test_predictions=evaluation(batch_size)
    plotting(y_test,test_predictions)
    results(y_test)
    
    
'''    
x_train = np.load(f'/home/hp/.config/spyder-py3/music-speech/wavfile/data/x_train.npy')
y_train = np.load(f'/home/hp/.config/spyder-py3/music-speech/wavfile/data/y_train.npy')
x_test = np.load(f'/home/hp/.config/spyder-py3/music-speech/wavfile/tdata/x_test.npy')
y_test = np.load(f'/home/hp/.config/spyder-py3/music-speech/wavfile/tdata/y_test.npy')

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

x_train = x_train/255
x_test = x_test/255

y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)

x_train, x_val, y_train_one_hot, y_val = train_test_split(x_train, y_train_one_hot, test_size=0.2)

model = Sequential()
model.add(Flatten())
model.add(Dense(32, input_dim=x_train.shape[1], activation='relu'))
model.add(Dense(16,  activation='relu'))
model.add(Dense(16,  activation='relu'))
model.add(Dense(3, activation='softmax'))  
model.compile(loss='categorical_crossentropy', optimizer='Adam',metrics=['accuracy'])

batch_size = 20
epochs = 20
history = model.fit(x_train, y_train_one_hot, epochs=epochs, batch_size = batch_size, validation_data=(x_val, y_val))
score = model.evaluate(x_test, y_test_one_hot, batch_size=batch_size)
summary=model.summary()
print(summary)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

test_predictions = model.predict(x_test)
print(test_predictions)
#print(np.argmax(test_predictions[1]))

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

predicted_classes = np.argmax(np.round(test_predictions),axis=1)
#print(predicted_classes)
#print(y_test.shape)
results=[]
labels=np.array([])
labels=y_test
results=np.equal(labels, predicted_classes)

import csv,os

header = 'Filename Test_Predictions Test_Predictions Test_Predictions Predicted_Class Ground_truth'
header = header.split()

output = open('FC_result.csv', 'w+', newline='') 
with output:
    writer = csv.writer(output)
    writer.writerow(header)
classes=['speech', 'music', 'm+s']
for c in classes:    
  for files in os.listdir(f'/home/hp/.config/spyder-py3/music-speech/wavfile/test/{c}'):
        file = f'/home/hp/.config/spyder-py3/music-speech/wavfile/test/{c}/{files}'
        to_append = f'{file}'
        #to_append += f' {c}'
        file = open('FC_result.csv', 'a')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())
            
with open("FC_result.csv","a" ,newline='') as my_csv:
    newarray = csv.writer(my_csv,delimiter=',')
    newarray.writerows(test_predictions)
    
import pandas as pd
result=pd.DataFrame({"class":predicted_classes, 
                      "labels":y_test, "Correction":results})
result.to_csv("FC_result.csv",mode='a',index=False)
'''