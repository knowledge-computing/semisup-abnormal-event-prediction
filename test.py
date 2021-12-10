import os
import logging
import pickle
import datetime

import sys 
sys.path.append("..") 

import numpy as np
import torch

from semi_supervised_AD.models.model_helper import init_model
from semi_supervised_AD.data_utils.ntt_dataset import NTTDataset
from semi_supervised_AD.data_utils.data_container import DataContainer
from semi_supervised_AD.utils.test_options import arg_parse, verbose
from semi_supervised_AD.trainer.tgconv_ngat_trainer import TGConvNGATTrainer
from semi_supervised_AD.utils.evaluate import evaluate
from semi_supervised_AD.models.others import svm_predict

        
if __name__ == '__main__':

    args = arg_parse()
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    
    ''' logging start '''
    log_file = os.path.join(args.res_path, f'{args.model_name}_test.log')
    logging.basicConfig(filename=log_file, level=logging.INFO, 
                        format='%(asctime)s %(message)s', datefmt='%m-%d %H:%M')
    verbose(args)
    
    ''' loading data '''
    dataset = NTTDataset(root=args.data_path)
    data = DataContainer(dataset, 
                         num_classes=args.num_classes,
                         batch_size=args.batch_size,
                         use_unlabel=args.use_unlabel,
                         shuffle=args.shuffle,
                         random_seed=args.random_seed,)
    
    model = init_model(data, args).to(device)
    model_path = os.path.join(args.res_path, 'models/model.pkl')
    model.load_state_dict(torch.load(model_path))
        
    trainer = TGConvNGATTrainer(data=data, model=model, 
                                device=device, 
                                eta=args.eta,
                                epochs=args.epochs,
                                lr=args.lr,
                                lr_milestones=(30, 80),
                                weight_decay=args.weight_decay,
                                patience=args.patience,
                                log_interval=args.log_interval,
                                predict_func=args.predict_func,
                                res_path=args.res_path)
    
    graph_emb, p, y = trainer.test(data.test_loader)
    
    if args.predict_func == 'SVM':
        clf = pickle.load(open(os.path.join(args.res_path, 'models/svm.pkl'), 'rb'))
        p = svm_predict(graph_emb, clf)
    
    print(p, y)
    acc = evaluate(p, y)
    trainer.test_info('Test', acc)

    