# -*- coding: utf-8 -*-
"""
Created on Thu May 26 17:19:53 2022

@author: marrd
I am using the 1600_10 dataset of 1600 points and 10 planned clusters.
I set eps = 1, minPts = 10 and Set number of minimum overlap here min_pts_overlap = 5
I then coded every point above 5 overlaps or more as core and all below as 
non-core in the class column. The model accuracy was between 0.8 and 1.0

"""
import os
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.metrics import plot_roc_curve
import matplotlib.pyplot as plt
from itertools import product
from sklearn import datasets
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.inspection import DecisionBoundaryDisplay

# Get current working directory
cwd = os.getcwd()
print(" Current working directory ", cwd)
# Change to ths directory
os.chdir(r"H:\GMU")
cwd = os.getcwd()
print(" Current working directory ", cwd)

# List file
for file in os.listdir(cwd):
    if file.endswith('.csv'):
        #print(file)        
        file_csv = cwd + "\\" + file
        print(file_csv)
        
in_file = r"H:\GMU\test_3.csv" 
csv_data = read_csv(in_file) 
print(csv_data.shape)
print(csv_data.head())

D = csv_data.values 

X = D[:,[0]]
y = D[:,3] # class column of core vs non-core 

# The values for # overlap, X , Y
x_tr, x_ts, y_tr, y_ts = train_test_split(X,y,test_size=0.20) 
model = SVC()
model.fit(x_tr, y_tr)
predict_threshold = model.predict(x_ts)
print("Accuracy: ", accuracy_score(y_ts, predict_threshold))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 32)

print("Number of observations and dimensions in training set:", X_train.shape)
print("Number of observations and dimensions in test set:", X_test.shape)
print("Number of observations in training set:", y_train.shape)
print("Number of observations in test set:", y_test.shape)

svmModel = SVC(random_state=1234, probability=True)
svmModel.fit(X_train, y_train)

y_pred = svmModel.predict(X_test)
print(y_pred)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred), ": is the confusion matrix")
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred), ": is the accuracy score")
from sklearn.metrics import precision_score
print(precision_score(y_test, y_pred), ": is the precision score")
from sklearn.metrics import recall_score
print(recall_score(y_test, y_pred), ": is the recall score")
from sklearn.metrics import f1_score
print(f1_score(y_test, y_pred), ": is the f1 score")