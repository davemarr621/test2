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

# The values for # overlap, X , Y
x = D[:,0:3]
y = D[:,3] # class column of core vs non-core  
x_tr, x_ts, y_tr, y_ts = train_test_split(x,y,test_size=20) 
model = SVC()
model.fit(x_tr, y_tr)
predict_threshold = model.predict(x_ts)
print("Accuracy: ", accuracy_score(y_ts, predict_threshold))

# Loading some example data
"""
iris = datasets.load_iris()
X = iris.data[:, [0, 2]]
y = iris.target
print(type(X))
print(X.shape)
print(X[:5, :])
print(type(y))
print(y.shape)
print(y)
"""

X = D[:,[1,2]]
y = D[:,3] # class column of core vs non-core 
# Training classifiers
clf1 = DecisionTreeClassifier(max_depth=4)
clf2 = KNeighborsClassifier(n_neighbors=7)
clf3 = SVC(gamma=0.1, kernel="rbf", probability=True)
eclf = VotingClassifier(
    estimators=[("dt", clf1), ("knn", clf2), ("svc", clf3)],
    voting="soft",
    weights=[2, 1, 2],
)

clf1.fit(X, y)
clf2.fit(X, y)
clf3.fit(X, y)
eclf.fit(X, y)

# Plotting decision regions
f, axarr = plt.subplots(2, 2, sharex="col", sharey="row", figsize=(10, 8))
for idx, clf, tt in zip(
    product([0, 1], [0, 1]),
    [clf1, clf2, clf3, eclf],
    ["Decision Tree (depth=4)", "KNN (k=7)", "Kernel SVM", "Soft Voting"],
):
    DecisionBoundaryDisplay.from_estimator(
        clf, X, alpha=0.4, ax=axarr[idx[0], idx[1]], response_method="predict"
    )
    axarr[idx[0], idx[1]].scatter(X[:, 0], X[:, 1], c=y, s=1)
    axarr[idx[0], idx[1]].set_title(tt)

plt.show()