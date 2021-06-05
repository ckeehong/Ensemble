# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 20:05:20 2021

@author: ckhon
"""
'''
https://machinelearningmastery.com/out-of-fold-predictions-in-machine-learning/
'''

### Out-of-Fold Predictions for Evaluation

# Approach 1: Estimate performance as the mean score estimated on each group of out-of-fold predictions.
# evaluate model by averaging performance across each fold
from numpy import mean
from numpy import std
from sklearn.datasets import make_blobs
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
# create the inputs and outputs
X, y = make_blobs(n_samples=1000, centers=2, n_features=100, cluster_std=20)
# k-fold cross validation
scores = list()
kfold = KFold(n_splits=10, shuffle=True)
# enumerate splits
for train_ix, test_ix in kfold.split(X):
	# get data
	train_X, test_X = X[train_ix], X[test_ix]
	train_y, test_y = y[train_ix], y[test_ix]
	# fit model
	model = KNeighborsClassifier()
	model.fit(train_X, train_y)
	# evaluate model
	yhat = model.predict(test_X)
	acc = accuracy_score(test_y, yhat)
	# store score
	scores.append(acc)
	print('> ', acc)
# summarize model performance
mean_s, std_s = mean(scores), std(scores)
print('Mean: %.3f, Standard Deviation: %.3f' % (mean_s, std_s))

# Approach 2: Estimate performance using the aggregate of all out-of-fold predictions.
# evaluate model by calculating the score across all predictions
from sklearn.datasets import make_blobs
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
# create the inputs and outputs
X, y = make_blobs(n_samples=1000, centers=2, n_features=100, cluster_std=20)
# k-fold cross validation
data_y, data_yhat = list(), list()
kfold = KFold(n_splits=10, shuffle=True)
# enumerate splits
for train_ix, test_ix in kfold.split(X):
	# get data
	train_X, test_X = X[train_ix], X[test_ix]
	train_y, test_y = y[train_ix], y[test_ix]
	# fit model
	model = KNeighborsClassifier()
	model.fit(train_X, train_y)
	# make predictions
	yhat = model.predict(test_X)
	# store
	data_y.extend(test_y)
	data_yhat.extend(yhat)
# evaluate the model
acc = accuracy_score(data_y, data_yhat)
print('Accuracy: %.3f' % (acc))


