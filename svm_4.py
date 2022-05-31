# -*- coding: utf-8 -*-
"""
Created on Fri May 27 15:31:49 2022
Differential evolution

@author: marrd
"""

from scipy import optimize
import os
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import scikitplot as skplt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import pandas as pd

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

D = csv_data.values # create pandas df
X = D[:,[0]]
y = D[:,3] # class column of core vs non-core 
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

X, y = make_classification(n_samples=1000, n_classes=2, random_state=1)

model = LogisticRegression()
model.fit(X, y)

# predict probabilities
preds = model.predict_proba(X)[:,1]
print(preds)

fpr, tpr, thresh = roc_curve(y, preds)

roc_df = pd.DataFrame(zip(fpr, tpr, thresh),columns = ["FPR","TPR","Threshold"])