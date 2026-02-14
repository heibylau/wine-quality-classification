import pandas as pd
import numpy as np
import sklearn.model_selection as skm
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score, classification_report

'''
This file contains functions that run the Logistic regression, 
Support Vector Machine, and Linear Discriminant Analysis.
'''

def _run_cv(model, x_train, y_train):
    '''
    Helper function to run a 5-fold cross validation.
    '''
    kfold = KFold(5, shuffle=True, random_state=42)
    cv_score = cross_val_score(model, x_train, y_train, cv=kfold)
    return cv_score

def _print_report(y_test, y_pred, cv_score):
    '''
    Helper function to print model performance.
    '''
    report = classification_report(y_test, y_pred, target_names=["0=Poor", "1=Good"])
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Average cross-validation score: {cv_score.mean():.2f}")
    print(f"Accuracy score: {accuracy:.2f}")

def _print_full_report(y_test, y_pred):
    print("="*60)
    report = classification_report(y_test, y_pred, target_names=["0=Poor", "1=Good"])
    print(report)

def run_logistic_regression(x_train, x_test, y_train, y_test):
    '''
    Trains and evaluates a logistic regression classifier.
    '''
    model = LogisticRegression(max_iter=1000)
    cv_score = _run_cv(model, x_train, y_train)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    _print_report(y_test, y_pred, cv_score)

def run_svm(x_train, x_test, y_train, y_test, kernel='linear', c=1, gamma=1, degree=2):
    '''
    Trains and evaluates a support vector classifier.
    '''
    model = SVC()
    if (kernel == 'poly'):
        model = SVC(kernel=kernel, C=c, gamma=gamma, degree=degree)
    elif (kernel == 'linear' or kernel == 'rbf'):
        model = SVC(kernel=kernel, C=c, gamma=gamma)
    cv_score = _run_cv(model, x_train, y_train)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    _print_report(y_test, y_pred, cv_score)

def run_lda(x_train, x_test, y_train, y_test, n_components=1):
    '''
    Trains and evaluates an LDA classifier.
    '''
    model = LinearDiscriminantAnalysis(n_components=n_components)
    cv_score = _run_cv(model, x_train, y_train)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    _print_report(y_test, y_pred, cv_score)