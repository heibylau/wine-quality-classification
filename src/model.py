import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, cross_val_score, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import learning_curve

'''
This file contains functions that run the Logistic regression, 
Support Vector Machine, and Linear Discriminant Analysis.
'''

def _run_cv(model, X_train, y_train):
    '''
    Helper function to run a 5-fold cross validation.
    '''
    kfold = KFold(5, shuffle=True, random_state=42)
    cv_score = cross_val_score(model, X_train, y_train, cv=kfold)
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

def run_logistic_regression(X_train, X_test, y_train, y_test):
    '''
    Trains and evaluates a logistic regression classifier.
    '''
    model = LogisticRegression(max_iter=1000)
    cv_score = _run_cv(model, X_train, y_train)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    _print_report(y_test, y_pred, cv_score)

def run_svm(X_train, X_test, y_train, y_test, kernel='linear', c=1, gamma=1, degree=2):
    '''
    Trains and evaluates a support vector classifier.
    '''
    model = SVC()
    if (kernel == 'poly'):
        model = SVC(kernel=kernel, C=c, gamma=gamma, degree=degree)
    elif (kernel == 'linear' or kernel == 'rbf'):
        model = SVC(kernel=kernel, C=c, gamma=gamma)
    cv_score = _run_cv(model, X_train, y_train)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    _print_report(y_test, y_pred, cv_score)

def run_lda(X_train, X_test, y_train, y_test, n_components=1):
    '''
    Trains and evaluates an LDA classifier.
    '''
    model = LinearDiscriminantAnalysis(n_components=n_components)
    cv_score = _run_cv(model, X_train, y_train)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    _print_report(y_test, y_pred, cv_score)

def tune_svm(X_train, y_train, kernel, c, gamma, degree=2):
    '''
    Tunes the hyperparameters for a support vector classifier.
    The hyperparameters are: cost, gamma, and degree (if kernel is polynomial)
    '''
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel=kernel))
    ])
    if kernel == 'rbf' or kernel == 'linear':
        param_dist = {
            'svm__C': c,
            'svm__gamma': gamma
        }
        search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_dist,
            n_iter=30,
            cv=5,
            verbose=1,
            n_jobs=-1,
            random_state=42
        )
    elif kernel == 'poly':
        param_dist = {
            'svm__C': c,
            'svm__gamma': gamma,
            'svm__degree': degree,
        }
        search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_dist,
            n_iter=15,
            cv=5,
            verbose=2,
            n_jobs=-1,
            random_state=42
        )

    search.fit(X_train, y_train)
    print(f"Best parameters for {kernel} kernel:", search.best_params_)
    return search.best_estimator_

def tune_lda(X_train, y_train):
    '''
    Tuning the LDA classifier.
    Solver: 'svd', 'lsqr', 'eigen'
    Shrinkage: None, 'auto
    '''
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('lda', LinearDiscriminantAnalysis())
    ])

    params = [
        {'lda__solver': ['svd']},  # svd cannot use shrinkage
        {'lda__solver': ['lsqr', 'eigen'], 'lda__shrinkage': [None, 'auto']}
    ]

    grid = GridSearchCV(pipeline, param_grid=params, cv=5, scoring='accuracy')

    grid.fit(X_train, y_train)
    print(f"Best parameters: {grid.best_params_}")
    return grid.best_estimator_

def build_classification_tree(X_train, y_train):
    '''
    Builds and tunes a classification tree.
    '''
    param_dist = {
        'max_depth': [None, 2, 3, 4, 5, 6, 7, 8, 10],
        'min_samples_split': [2, 5, 10, 15],
        'min_samples_leaf': [1, 2, 4, 6],
        'criterion': ['gini', 'entropy']
    }

    tree = DecisionTreeClassifier(random_state=42)

    search = RandomizedSearchCV(
        tree,
        param_distributions=param_dist,
        n_iter=20,
        cv=5,
        scoring='accuracy',
        verbose=1,
        n_jobs=-1,
        random_state=42
    )

    search.fit(X_train, y_train)
    print(f"Best parameters: {search.best_params_}")
    print(f"Best CV score: {search.best_score_:.2f}")
    return search.best_estimator_

def plot_learning_curve(model, X_train, y_train, title="Learning Curve"):
    
    train_sizes, train_scores, val_scores = learning_curve(
        model,
        X_train,
        y_train,
        cv=5,
        scoring="accuracy",
        train_sizes=np.linspace(0.1, 1.0, 10),
        shuffle=True,
        random_state=42,
        n_jobs=-1
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    plt.figure(figsize=(8,6))
    plt.plot(train_sizes, train_mean, 'o-', label="Training Accuracy")
    plt.plot(train_sizes, val_mean, 'o-', label="Validation Accuracy")

    plt.fill_between(train_sizes,
                     train_mean - train_std,
                     train_mean + train_std,
                     alpha=0.1)

    plt.fill_between(train_sizes,
                     val_mean - val_std,
                     val_mean + val_std,
                     alpha=0.1)

    plt.xlabel("Training Set Size")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()