import os
import shutil
import logging
import pickle
import datetime

import sys 
sys.path.append("..") 

import numpy as np
import torch

from utils.train_options import parse_args, load_param_dict
from data_utils.ntt_dataset import NTTDataset
from data_utils.data_loader import NTTDataLoader
from models.model_loader import load_model

from models.tgconv_ngat_ae import TGConvNGATAE
from trainers.trainer_loader import load_trainer


def init_weights_from_pretraining(model, pretrain_model_name, device):
    
    pretrain_model_path = os.path.join(params['res_path'], '../', 
                                       pretrain_model_name, 'models/model.pkl')

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

        
def train(params):

    device = torch.device('cuda:{}'.format(params['gpu_id']) if torch.cuda.is_available() else 'cpu')
    logging.info(f'Device - GPU:{device}')
        
    ''' loading data '''
    if params['reload_data']:
        processed_path = os.path.join(params['data_path'], f'processed')
        if os.path.exists(processed_path):
            shutil.rmtree(processed_path)

    dataset = NTTDataset(root=params['data_path'])
    data = NTTDataLoader(dataset, 
                         num_classes=params['num_classes'],
                         batch_size=params['batch_size'],
                         use_unlabel=params['use_unlabel'],
                         shuffle=params['shuffle'],
                         random_seed=params['random_seed'],)
    
    model = load_model(data, params).to(device)
    if params['use_pretrain']:
        model = init_weights_from_pretraining(model, params['pretrain_model_name'], device)
    
    trainer = load_trainer(data, model, device, params)
    trainer.train()
    


def main():
    
    args = parse_args()
    params = load_param_dict(args, mode='train')
    train(params)
    
    
if __name__ == "__main__":
    main()
