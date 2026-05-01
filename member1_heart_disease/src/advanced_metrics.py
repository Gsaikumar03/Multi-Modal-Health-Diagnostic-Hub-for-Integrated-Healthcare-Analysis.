from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve,
    precision_recall_curve, auc
)
import numpy as np


def compute_all_metrics(y_true, y_pred, y_prob):

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "specificity": specificity
    }

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)

    roc_auc = auc(fpr, tpr)
    pr_auc = auc(recall_curve, precision_curve)

    return metrics, cm, fpr, tpr, roc_auc, precision_curve, recall_curve, pr_auc