import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn import metrics

from anomaly_detection.models.others import svm_classify, svm_regression, svm_predict


def one_hot_encoding(labels): 
    targets = torch.zeros(labels.size(0), 2)
    for i, label in enumerate(labels):
        targets[i, label] = 1
    return targets


def test(m, loader, device):
    """ test the model with given data
    Return
        l_enc: embeddings for nodes
        g_enc: embeddings for graphs
        y: labels for graphs
        out: predictions for graphs
        node_y: labels for nodes
    """

    l_enc, g_enc, y, pred, node_y = [], [], [], [], []

    m.eval()
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            l, g = m.encoder(data)
            p = F.log_softmax(m.predict(g), dim=1)
            l_enc.append(l.cpu().numpy())
            g_enc.append(g.cpu().numpy())
            y.append(data.y.cpu().numpy())
            pred.append(p.cpu().numpy())
            node_y.append(data.node_label.cpu().numpy())

    l_enc = np.concatenate(l_enc, axis=0)
    g_enc = np.concatenate(g_enc, axis=0)
    y = np.concatenate(y, axis=0).reshape(-1)
    pred = np.argmax(np.concatenate(pred, axis=0), axis=1)
    node_y = np.concatenate(node_y, axis=0)

    return l_enc, g_enc, y, pred, node_y


def evaluate(y, p):
    """ return precision 0 recall 0 precision 1 recall 1 and overall f1 """

    y, p = y[(y == 0) | (y == 1)], p[(y == 0) | (y == 1)]
    prec = precision_score(y, p, average=None, zero_division=0)
    recall = recall_score(y, p, average=None, zero_division=0)
    score = f1_score(y, p)
    return prec[0], recall[0], prec[1], recall[1], score


def svm_classify_and_predict(train_x, train_y, val_x, test_x, label_weight):
    
    train_y_mask = train_y >= 0
    clf = svm_classify(train_x[train_y_mask], train_y[train_y_mask], label_weight)
    train_p = svm_predict(train_x, clf)
    val_p = svm_predict(val_x, clf)
    test_p = svm_predict(test_x, clf)
    res = {'train_p': train_p, 'val_p': val_p, 'test_p': test_p}
    return clf, res


def test_and_evaluate(m, train_loader, val_loader, test_loader, label_weight, device):
    _, train_g_enc, train_y, _, _ = test(m, train_loader, device)
    _, val_g_enc, val_y, _, _ = test(m, val_loader, device)
    _, test_g_enc, test_y, _, _ = test(m, test_loader, device)
    _, res = svm_classify_and_predict(train_g_enc, train_y, val_g_enc, test_g_enc, label_weight)
    val_loss = evaluate(val_y, res['val_p'])
    test_loss = evaluate(test_y, res['test_p'])
    return val_loss, test_loss

