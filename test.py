import os
import logging
import pickle
import datetime

import sys 
sys.path.append("..") 

import numpy as np
import torch

from models.model_loader import load_model
from data_utils.ntt_dataset import NTTDataset
from data_utils.data_loader import NTTDataLoader
from utils.test_options import parse_args, load_param_dict
from trainer.tgconv_ngat_trainer import TGConvNGATTrainer
from utils.evaluate import evaluate
from models.others import svm_predict

        
if __name__ == '__main__':

    args = parse_args()
    params = load_param_dict(args, mode='test')
    
    device = torch.device('cuda:{}'.format(params['gpu_id']) if torch.cuda.is_available() else 'cpu')
        
    ''' loading data '''
    dataset = NTTDataset(root=params['data_path'])
    data = NTTDataLoader(dataset, 
                         num_classes=params['num_classes'],
                         batch_size=params['batch_size'],
                         use_unlabel=params['use_unlabel'],
                         shuffle=params['shuffle'],
                         random_seed=params['random_seed'],)
    
    model = load_model(data, params).to(device)
    model_path = os.path.join(params['res_path'], 'models/model.pkl')
    model.load_state_dict(torch.load(model_path))
        
    trainer = TGConvNGATTrainer(data=data, 
                                model=model, 
                                device=device, 
                                eta=params['eta'],
                                epochs=params['epochs'],
                                lr=params['lr,']
                                lr_milestones=(30, 80),
                                weight_decay=params['weight_decay'],
                                patience=params['patience'],
                                predict_func=params['predict_func'],
                                res_path=params['res_path'])
    
    graph_emb, p, y = trainer.test(data.test_loader)
    
    if args.predict_func == 'SVM':
        clf = pickle.load(open(os.path.join(params['res_path'], 'models/svm.pkl'), 'rb'))
        p = svm_predict(graph_emb, clf)
    
    print(p, y)
    acc = evaluate(p, y)
    trainer.test_info('Test', acc)

    