import os
import logging
import pickle
import datetime

import sys 
sys.path.append("..") 

import numpy as np
import torch

from semi_supervised_AD.models.tgconv_ngat_ae import TGConvNGATAE
from semi_supervised_AD.models.model_helper import init_model
from semi_supervised_AD.data_utils.ntt_dataset import NTTDataset
from semi_supervised_AD.data_utils.data_container import DataContainer
from semi_supervised_AD.utils.train_options import arg_parse, verbose, initialize_tb
from semi_supervised_AD.trainer.tgconv_ngat_trainer import TGConvNGATTrainer
from semi_supervised_AD.trainer.tgconv_ngat_ae_trainer import TGConvNGATAETrainer
from semi_supervised_AD.trainer.tgconv_ngat_cc_trainer import TGConvNGATCCTrainer
from semi_supervised_AD.trainer.tgconv_ngat_cc1_trainer import TGConvNGATCC1Trainer
from semi_supervised_AD.trainer.tgconv_ngat_cluster_trainer import TGConvNGATClusterTrainer
from semi_supervised_AD.trainer.tgconv_ngat_cluster_trainer1 import TGConvNGATClusterTrainer1
from semi_supervised_AD.trainer.tgconv_ngat_cluster_trainer2 import TGConvNGATClusterTrainer2

def init_weights_from_pretraining(model, pretrain_model_name, device):
    
    pretrain_model_path = os.path.join(args.res_path, '../', pretrain_model_name, 'models/model.pkl')

    if not os.path.exists(pretrain_model_path):
        print('The pretrain model path does not exists.')
        return model
        
    pretrain_model_dict = torch.load(pretrain_model_path, map_location=device)
    model_dict = model.state_dict()

    # filter out decoder network keys
    pretrain_model_dict = {k: v for k, v in pretrain_model_dict.items() if k in model_dict}
    model_dict.update(pretrain_model_dict)
    model.load_state_dict(model_dict)
    print('The model has been initialized with the pretrain model.')
    return model

        
if __name__ == '__main__':

    args = arg_parse()
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    
    ''' logging start '''
    log_file = os.path.join(args.res_path, f'{args.model_name}.log')
    logging.basicConfig(filename=log_file, level=logging.INFO, 
                        format='%(asctime)s %(message)s', datefmt='%m-%d %H:%M')
    tb_writer = initialize_tb(args)    
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
    
    if args.use_pretrain:
        model = init_weights_from_pretraining(model, args.pretrain_model_name, device)
    
    net_name = args.model_name.split('_')[0]
    
    if net_name == 'TGConvNGATAE':
        trainer = TGConvNGATAETrainer(data=data, model=model, 
                                      device=device, 
                                      epochs=args.epochs,
                                      lr=args.lr,
                                      lr_milestones=(40,80),
                                      weight_decay=args.weight_decay,
                                      res_path=args.res_path,
                                      verbose=args.verbose)
        
    elif net_name == 'TGConvNGAT':
        trainer = TGConvNGATTrainer(data=data, model=model, 
                                    device=device, 
                                    eta=args.eta,
                                    epochs=args.epochs,
                                    lr=args.lr,
                                    weight_decay=args.weight_decay,
                                    patience=args.patience,
                                    log_interval=args.log_interval,
                                    predict_func=args.predict_func,
                                    res_path=args.res_path,
                                    verbose=args.verbose)         
        
    elif net_name == 'TGConvNGATCC':
        trainer = TGConvNGATCCTrainer(data=data, model=model, 
                                      device=device, 
                                      epochs=args.epochs,
                                      lr=args.lr,
                                      lr_center=args.lr_center,
                                      weight_decay=args.weight_decay,
                                      patience=args.patience,
                                      log_interval=args.log_interval,
                                      predict_func=args.predict_func,
                                      res_path=args.res_path,
                                      verbose=args.verbose)        
        
    elif net_name == 'TGConvNGATCC1':
        trainer = TGConvNGATCC1Trainer(data=data, model=model, 
                                      device=device, 
                                      epochs=args.epochs,
                                      lr=args.lr,
                                      lr_center=args.lr_center,
                                      weight_decay=args.weight_decay,
                                      patience=args.patience,
                                      log_interval=args.log_interval,
                                      predict_func=args.predict_func,
                                      res_path=args.res_path,
                                      verbose=args.verbose)       
        
    elif net_name == 'TGConvNGATCluster':
        trainer = TGConvNGATClusterTrainer(data=data, model=model, 
                                           device=device, 
                                           epochs=args.epochs,
                                           lr=args.lr,
                                           lr_center=args.lr_center,
                                           weight_decay=args.weight_decay,
                                           patience=args.patience,
                                           log_interval=args.log_interval,
                                           predict_func=args.predict_func,
                                           res_path=args.res_path,
                                           verbose=args.verbose)       
    elif net_name == 'TGConvNGATCluster1':
        trainer = TGConvNGATClusterTrainer1(data=data, model=model, 
                                            device=device, 
                                            epochs=args.epochs,
                                            lr=args.lr,
                                            lr_center=args.lr_center,
                                            weight_decay=args.weight_decay,
                                            patience=args.patience,
                                            log_interval=args.log_interval,
                                            predict_func=args.predict_func,
                                            res_path=args.res_path,
                                            verbose=args.verbose)     
    elif net_name == 'TGConvNGATCluster2':
        trainer = TGConvNGATClusterTrainer2(data=data, model=model, 
                                            device=device, 
                                            epochs=args.epochs,
                                            lr=args.lr,
                                            lr_center=args.lr_center,
                                            weight_decay=args.weight_decay,
                                            patience=args.patience,
                                            log_interval=args.log_interval,
                                            predict_func=args.predict_func,
                                            res_path=args.res_path,
                                            verbose=args.verbose)     
    trainer.train()
    