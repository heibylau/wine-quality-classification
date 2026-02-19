import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import learning_curve
from sklearn.metrics import confusion_matrix, roc_curve, auc

'''
This file contains functions that visualizes the performance of fitted models.
'''

def plot_accuracy_chart(models, model_names, X_test, y_test):
    """
    Plots accuracy bar chart.
    """
    accuracies = [model.score(X_test, y_test) for model in models]
    
    plt.figure(figsize=(8, 5))
    bars = plt.bar(model_names, accuracies, color=sns.color_palette("tab10", len(models)))

    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{acc:.3f}", 
                 ha='center', va='bottom', fontsize=10)
    
    plt.xlabel("Models")
    plt.xticks(rotation=45)
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy Comparison between the Top 4 Models")
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0)
    plt.show()


def plot_confusion_matrices(models, model_names, X_test, y_test, class_names=["Poor","Good"]):
    """
    Plots confusion matrices for the top 4 models in a 2x2 grid.
    """
    n_models = len(models)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i in range(n_models):
        y_pred = models[i].predict(X_test)  
        cm = confusion_matrix(y_test, y_pred)  
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', 
                    xticklabels=class_names, yticklabels=class_names,
                    ax=axes[i], cbar=False)
        
        axes[i].set_title(model_names[i])
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("True")
    fig.suptitle("Confusion Matrices for the Top 4 Models", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_learning_curves(models, model_names, X_train, y_train):
    """
    Plots learning curves for the top 4 models in a 2x2 grid.
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten() 
    
    for i, model in enumerate(models):
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

        ax = axes[i]
        ax.plot(train_sizes, train_mean, 'o-', label="Training Accuracy")
        ax.plot(train_sizes, val_mean, 'o-', label="Validation Accuracy")
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
        ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1)
        ax.set_xlabel("Training Set Size")
        ax.set_ylabel("Accuracy")
        ax.set_title(model_names[i])
        ax.grid(True)
        ax.legend()

    fig.suptitle("Learning Curves for the Top 4 Models", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_roc_curves(models, model_names, X_test, y_test):
    """
    Plots ROC curves for the top 4 models.
    """

    plt.figure(figsize=(8, 6))
    
    for model, name in zip(models, model_names):
        if hasattr(model, "decision_function"):
            y_score = model.decision_function(X_test)
        else:
            y_score = model.predict_proba(X_test)[:, 1]
        
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc:.3f})")
    
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves for the Top 4 Models")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()