import numpy as np

from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    f1_score,
    precision_score, 
    recall_score, 
    balanced_accuracy_score, 
    roc_curve, 
    auc
)

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Calculate macro F1-score, precision, recall, and balanced accuracy.
    
    Parameters:
    y_true (array-like): True labels.
    y_pred (array-like): Predicted labels.
    
    Returns:
    dict: Dictionary containing the metrics.
    """
    metrics = {
        'macro_f1': f1_score(y_true, y_pred, average='macro'),
        'macro_precision': precision_score(y_true, y_pred, average='macro'),
        'macro_recall': recall_score(y_true, y_pred, average='macro'),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred)
    }
    return metrics

def calculate_multiclass_roc_auc(y_true, y_proba, classes):
    """
    Calculate ROC curve and AUC for each class in a multi-class setting.
    
    Parameters:
    y_true (array-like): True labels.
    y_proba (array-like): Predicted probabilities, shape (n_samples, n_classes).
    classes (array-like): List of all classes.
    
    Returns:
    dict: Dictionary containing the ROC curve and AUC for each class.
    """
    # Binarize the output
    y_true_bin = label_binarize(y_true, classes=classes)
    
    roc_auc_dict = {}
    
    for i, class_label in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        
        roc_auc_dict[class_label] = {
            'fpr': fpr,
            'tpr': tpr,
            'roc_auc': roc_auc
        }
    
    return roc_auc_dict