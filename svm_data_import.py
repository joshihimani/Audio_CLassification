#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 15:34:02 2021

@author: hp
"""



import os
import librosa
import librosa.display
import numpy as np
import csv 

header = 'mfcc'
for i in range(1, 20):
    header += f' mfcc{i}'
header += ' label'
header = header.split()

file = open('dataset.csv', 'w', newline='')
with file:
    writer = csv.writer(file)
    writer.writerow(header)
classes='speech music m+s'.split()
for c in classes:    
  for files in os.listdir(f'/home/hp/.config/spyder-py3/music-speech/wavfile/train/{c}'):
        file = f'/home/hp/.config/spyder-py3/music-speech/wavfile/train/{c}/{files}'
        y, sr = librosa.load(file)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        to_append = f'{file}'
        for e in mfcc:
            to_append += f' {np.mean(e)}'
        to_append += f' {c}'
        file = open('dataset.csv', 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())
            
header = 'mfcc'
for i in range(1, 20):
    header += f' mfcc{i}'
header += ' label'
header = header.split()

file = open('dataset1.csv', 'w', newline='')
with file:
    writer = csv.writer(file)
    writer.writerow(header)
classes='speech music m+s'.split()
for c in classes:    
  for files in os.listdir(f'/home/hp/.config/spyder-py3/music-speech/wavfile/test/{c}'):
        file = f'/home/hp/.config/spyder-py3/music-speech/wavfile/test/{c}/{files}'
        y, sr = librosa.load(file)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        to_append = f'{file}'
        for e in mfcc:
            to_append += f' {np.mean(e)}'
        to_append += f' {c}'
        file = open('dataset1.csv', 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())
            
