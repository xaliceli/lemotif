"""
eval.py
Evaluation functions
"""

import numpy as np

def f1(true, pred):
    pred_positive_correct = np.sum(np.logical_and(true == pred, pred == 1))
    pred_positive_all = np.sum(pred == 1)
    true_positive_all = np.sum(true == 1)

    prec = pred_positive_correct/pred_positive_all
    rec = pred_positive_correct/true_positive_all

    return np.mean(2*prec*rec/(prec+rec))

def precision(true, pred):
    pred_positive_correct = np.sum(np.logical_and(true == pred, pred == 1))
    pred_positive_all = np.sum(pred == 1)

    return np.mean(pred_positive_correct/pred_positive_all)

def recall(true, pred):
    pred_positive_correct = np.sum(np.logical_and(true == pred, pred == 1))
    true_positive_all = np.sum(true == 1)

    return np.mean(pred_positive_correct/true_positive_all)

def norm_acc(true, pred):
    pred_positive_correct = np.sum(np.logical_and(true == pred, pred == 1))
    true_positive_all = np.sum(true == 1)

    pred_negative_correct = np.sum(np.logical_and(true == pred, pred == 0))
    true_negative_all = np.sum(true == 0)

    return np.mean((pred_positive_correct/true_positive_all + pred_negative_correct/true_negative_all)/2)