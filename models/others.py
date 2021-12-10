import torch
import torch.nn as nn
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

    
def svm_classify(train_x, train_y, label_weight=None):

    if label_weight is not None:
        class_weight = {0: label_weight[0], 1: label_weight[1]}
    else:
        class_weight = None
            
    clf = make_pipeline(StandardScaler(), 
                        SVC(gamma='auto', class_weight=class_weight, random_state=1234))
    clf.fit(train_x, train_y)
    return clf


def svm_regression(train_x, train_y):

    clf = make_pipeline(StandardScaler(), SVR(gamma='auto'))
    clf.fit(train_x, train_y)
    return clf


def svm_predict(test_x, clf):
    return clf.predict(test_x)

