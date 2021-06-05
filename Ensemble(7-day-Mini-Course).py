# -*- coding: utf-8 -*-
"""
Created on Sat May  8 15:01:12 2021

@author: ckhon
"""
'''
https://machinelearningmastery.com/ensemble-machine-learning-with-python-7-day-mini-course/?fbclid=IwAR2o63X44KqQ89QwoyFjXFs8EI3CGOxuBwT7OTjPxY94Numz3kXTBiQc0OE
'''

### Lesson 01: What Is Ensemble Learning?

'''
Bagging, e.g. bagged decision trees and random forest.
Boosting, e.g. adaboost and gradient boosting
Stacking, e.g. voting and using a meta-model.
'''

### Lesson 02: Bagging Ensembles

# example of evaluating a bagging ensemble for classification
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import BaggingClassifier
# create the synthetic classification dataset
X, y = make_classification(random_state=1)
# configure the ensemble model
model = BaggingClassifier(n_estimators=50)
# configure the resampling method
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate the ensemble on the dataset using the resampling method
n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# report ensemble performance
print('Mean Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
'''
For bonus points, evaluate the effect of using more decision trees in the ensemble or 
even change the base learner that is used.
'''

### Lesson 03: Random Forest Ensemble

# example of evaluating a random forest ensemble for classification
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
# create the synthetic classification dataset
X, y = make_classification(random_state=1)
# configure the ensemble model
model = RandomForestClassifier(n_estimators=50)
# configure the resampling method
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate the ensemble on the dataset using the resampling method
n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# report ensemble performance
print('Mean Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
'''
For bonus points, evaluate the effect of using more decision trees in the ensemble or 
tuning the number of features to consider at each split point.
'''

### Lesson 04: AdaBoost Ensemble

# example of evaluating an adaboost ensemble for classification
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import AdaBoostClassifier
# create the synthetic classification dataset
X, y = make_classification(random_state=1)
# configure the ensemble model
model = AdaBoostClassifier(n_estimators=50)
# configure the resampling method
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate the ensemble on the dataset using the resampling method
n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# report ensemble performance
print('Mean Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
'''
For bonus points, evaluate the effect of using more decision trees in the ensemble or 
even change the base learner that is used (note, it must support weighted training data).
'''

### Lesson 05: Gradient Boosting Ensemble

# example of evaluating a gradient boosting ensemble for classification
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
# create the synthetic classification dataset
X, y = make_classification(random_state=1)
# configure the ensemble model
model = GradientBoostingClassifier(n_estimators=50)
# configure the resampling method
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate the ensemble on the dataset using the resampling method
n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# report ensemble performance
print('Mean Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
'''
For bonus points, evaluate the effect of using more decision trees in the ensemble or 
try different learning rate values.
'''

### Lesson 06: Voting Ensemble

# example of evaluating a voting ensemble for classification
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
# create the synthetic classification dataset
X, y = make_classification(random_state=1)
# configure the models to use in the ensemble
models = [('lr', LogisticRegression()), ('nb', GaussianNB())]
# configure the ensemble model
model = VotingClassifier(models, voting='soft')
# configure the resampling method
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate the ensemble on the dataset using the resampling method
n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# report ensemble performance
print('Mean Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
'''
For bonus points, evaluate the effect of trying different types of models in the ensemble or 
even change the type of voting from soft voting to hard voting.
'''

### Lesson 07: Stacking Ensemble

# example of evaluating a stacking ensemble for classification
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
# create the synthetic classification dataset
X, y = make_classification(random_state=1)
# configure the models to use in the ensemble
models = [('knn', KNeighborsClassifier()), ('tree', DecisionTreeClassifier())]
# configure the ensemble model
model = StackingClassifier(models, final_estimator=LogisticRegression(), cv=3)
# configure the resampling method
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate the ensemble on the dataset using the resampling method
n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# report ensemble performance
print('Mean Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
'''
For bonus points, evaluate the effect of trying different types of models in the ensemble 
and different meta-models to combine the predictions.
'''

