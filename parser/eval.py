"""
eval.py
Evaluation functions
"""

import numpy as np

def f1(true, pred):
    prec, rec = precision(true, pred), recall(true, pred)
    return 2*prec*rec/(prec+rec)

def precision(true, pred):
    pred_positive_correct = np.sum(np.logical_and(true == pred, pred == 1))
    pred_positive_all = np.sum(pred == 1)

    return pred_positive_correct/pred_positive_all

def recall(true, pred):
    pred_positive_correct = np.sum(np.logical_and(true == pred, pred == 1))
    true_positive_all = np.sum(true == 1)

    return pred_positive_correct/true_positive_all