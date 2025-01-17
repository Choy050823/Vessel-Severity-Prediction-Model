import numpy as np
from scipy.stats import mode
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

def majority_voting(predictions):
    """Combine predictions using majority voting."""
    ensemble_predictions, _ = mode(predictions, axis=0)
    return ensemble_predictions.flatten()

def meta_learner(predictions, labels):
    """Train a meta-learner (logistic regression) on the predictions."""
    meta_learner = LogisticRegression()
    meta_learner.fit(predictions.T, labels)
    return meta_learner

def evaluate_ensemble(y_true, y_pred):
    """Evaluate the ensemble's performance."""
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Ensemble Accuracy: {accuracy:.4f}")