import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, cross_val_score, KFold
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report


'''
This file contains functions that run the Logistic regression, 
Support Vector Machine, Linear Discriminant Analysis, and Classification Decision Tree models.
'''


# -----The following functions are for running models without tuning-----
def run_logistic_regression(X_train, X_test, y_train, y_test):
    '''
    Trains and evaluates a logistic regression classifier.
    '''
    model = LogisticRegression()
    cv_score = _run_cv(model, X_train, y_train)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    _print_report(y_test, y_pred, cv_score)
    return model


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
    return model


def run_lda(X_train, X_test, y_train, y_test, n_components=1):
    '''
    Trains and evaluates an LDA classifier.
    '''
    model = LinearDiscriminantAnalysis(n_components=n_components)
    cv_score = _run_cv(model, X_train, y_train)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    _print_report(y_test, y_pred, cv_score)
    return model


def run_classification_tree(X_train, X_test, y_train, y_test):
    '''
    Trains and evaluates a Classification tree.
    '''
    tree = DecisionTreeClassifier(random_state=42)
    cv_score = _run_cv(tree, X_train, y_train)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    _print_report(y_test, y_pred, cv_score)
    return tree


# -----The following functions are for hyperparameter tuning-----
def tune_logistic_regression(X_train, y_train):
    """
    Tune Logistic Regression using C and l1_ratio for regularization.
    - l1_ratio = 0 → L2 only
    - l1_ratio = 1 → L1 only
    - 0 < l1_ratio < 1 → ElasticNet
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('logistic', LogisticRegression(solver='saga'))
    ])
    
    param_grid = {
        'logistic__C': [0.01, 0.1, 1, 10, 100],
        'logistic__l1_ratio': [0, 0.25, 0.5, 0.75, 1] 
    }
    
    grid = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    grid.fit(X_train, y_train)
    
    print("Best parameters:", grid.best_params_)
    return grid.best_estimator_


def tune_svm(X_train, y_train, kernel, c, gamma, degree=2, n_iter=30):
    '''
    Tunes SVM using cost, gamma, and degree(if kernel is polynomial).
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
            n_iter=n_iter,
            cv=5,
            verbose=1,
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
    Shrinkage: None, 'auto'
    '''
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('lda', LinearDiscriminantAnalysis(n_components=1))
    ])

    params = [
        {'lda__solver': ['svd']}, 
        {'lda__solver': ['lsqr', 'eigen'], 'lda__shrinkage': [None, 'auto']}
    ]

    grid = GridSearchCV(pipeline, param_grid=params, cv=5, scoring='accuracy')

    grid.fit(X_train, y_train)
    print(f"Best parameters: {grid.best_params_}")
    return grid.best_estimator_


def tune_classification_tree(X_train, y_train):
    '''
    Tunes a classification tree using max_depth, min_samples_split, min_samples_leaf, and criterion.
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


# -----The following functions are for visualization-----
def show_tree(tree):
    """
    Visualizes a decision tree.
    """
    plt.figure(figsize=(20,10))
    plot_tree(
        tree, 
        feature_names=['is_red_wine', 'alcohol', 'density', 'volatile_acidity', 'chlorides'],
        class_names=['Poor','Good'],
        filled=True,
        rounded=True,
        fontsize=10
    )
    plt.show()


def print_classification_report(models, model_names, X_test, y_test):
    """
    Prints classification report for each model.
    """
    for i, model in enumerate(models):
        print(f"Classification Report for {model_names[i]}:")
        y_pred = model.predict(X_test)
        print(classification_report(y_test, y_pred, target_names=["0=Poor", "1=Good"]))
        print("="*80)


# -----The following are helper functions to be used within this file only-----
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
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Average cross-validation score: {cv_score.mean()}")
    print(f"Accuracy score: {accuracy}")