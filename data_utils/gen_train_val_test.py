import sys
import numpy as np
import random
import logging


def split_train_val_test(dataset, num_classes=2, shuffle=True, use_unlabel=False, random_seed=1234):
    """ construct train, val, test datasets """
    
    classes = [i for i in range(num_classes)]
    sup_index_dict, unsup_index = {i: [] for i in classes}, []
    for i, data in enumerate(dataset):
        label = data.y.numpy()[0]
        if label in classes:
            sup_index_dict[label].append(i)
        else:
            unsup_index.append(i)
    
    def construct_train_val_test(idx):
        num_data = len(idx)
        num_train, num_val, num_test = int(0.6 * num_data), int(0.2 * num_data), int(0.2 * num_data)
        train_idx = idx[:num_train]
        val_idx = idx[num_train: (num_train + num_val)]
        test_idx = idx[(num_train + num_val):]
        return train_idx, val_idx, test_idx

    if shuffle:
        random.seed(random_seed)
        for _, v in sup_index_dict.items():
            random.shuffle(v)

    train_index, val_index, test_index = [], [], []
    for _, v in sup_index_dict.items():
        train, val, test = construct_train_val_test(v)
        train_index += train
        val_index += val
        test_index += test
    
    train_index = [i for i in train_index if 1 < i < len(dataset) - 1]    
    train_dataset = dataset.index_select(idx=train_index)
    val_dataset = dataset.index_select(idx=val_index)
    test_dataset = dataset.index_select(idx=test_index)

    if len(unsup_index) == 0 and use_unlabel:
        logging.info('There is no unlabeled data in the dataset.')

    if len(unsup_index) > 0 and use_unlabel:
        unsup_dataset = dataset.index_select(idx=unsup_index)
        logging.info('num_train: {} [{}/{}], num_val: {}, num_test: {}.'
                     .format(len(train_index)+len(unsup_index), len(train_index), len(unsup_index), len(val_dataset), len(test_dataset)))
    else:
        unsup_dataset = None
        print('num_train: {}, num_val: {}, num_test: {}.'
                     .format(len(train_dataset), len(val_dataset), len(test_dataset)))
        logging.info('num_train: {}, num_val: {}, num_test: {}.'
                     .format(len(train_dataset), len(val_dataset), len(test_dataset)))

    return train_dataset, val_dataset, test_dataset, unsup_dataset


def compute_label_weight(dataset, num_classes):
    """ compute the weights for labels
    Params
        data: NTT dataset
    Return
        weights: an array([weight1, weight2, weight3, ....])
    """

    label_dict = dict()
    for i, data in enumerate(dataset):
        label = data.y.numpy()[0]
        if label_dict.get(label) is None:
            label_dict[label] = 0
        label_dict[label] += 1

    num_samples_per_cls = [label_dict[i] for i in range(num_classes)]
    total_num_samples = sum(num_samples_per_cls)

    beta = 0.9999
    norm_samples_per_cls = np.array([i / total_num_samples for i in num_samples_per_cls])
    effective_num = 1. - np.power(beta, norm_samples_per_cls)
    weights = (1. - beta) / np.array(effective_num)
    weights = weights / np.sum(weights)
    
    print('weighted labels: {}'.format(weights))
    logging.info('weighted labels: {}'.format(weights))
    return weights
