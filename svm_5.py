# -*- coding: utf-8 -*-
"""
Created on Tue May 31 12:26:01 2022

@author: marrd
"""

# search thresholds for imbalanced classification
import os
from pandas import read_csv
from numpy import arange
from numpy import argmax
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# Get current working directory
cwd = os.getcwd()
print(" Current working directory ", cwd)
# Change to ths directory
os.chdir(r"H:\GMU")
cwd = os.getcwd()
print(" Current working directory ", cwd)

in_file = r"H:\GMU\test_3.csv" 
csv_data = read_csv(in_file) 
print(csv_data.shape)
print(csv_data.head())

D = csv_data.values # create pandas df
X = D[:,[0]]
y = D[:,3] # class column of core vs non-core 

# apply threshold to positive probabilities to create labels
def to_labels(pos_probs, threshold):
	return (pos_probs >= threshold).astype('int')

trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2, stratify=y)
# fit a model
model = LogisticRegression(solver='lbfgs')
model.fit(trainX, trainy)
# predict probabilities
yhat = model.predict_proba(testX)
# keep probabilities for the positive outcome only
probs = yhat[:, 1]
# define thresholds
thresholds = arange(0, 1, 0.001)
# evaluate each threshold
scores = [f1_score(testy, to_labels(probs, t)) for t in thresholds]
# get best threshold
ix = argmax(scores)
print('Threshold=%.3f, F-Score=%.5f' % (thresholds[ix], scores[ix]))