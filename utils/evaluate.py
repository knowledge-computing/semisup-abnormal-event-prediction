import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn import metrics


def one_hot_encoding(labels): 
    targets = torch.zeros(labels.size(0), 2)
    for i, label in enumerate(labels):
        targets[i, label] = 1
    return targets


def evaluate(y, p):
    """ return precision 0 recall 0 precision 1 recall 1 and overall f1 """

    y, p = y[(y == 0) | (y == 1)], p[(y == 0) | (y == 1)]
    prec = precision_score(y, p, average=None, zero_division=0)
    recall = recall_score(y, p, average=None, zero_division=0)
    score = f1_score(y, p)
    return prec[0], recall[0], prec[1], recall[1], score

