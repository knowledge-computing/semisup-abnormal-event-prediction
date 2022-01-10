import os
import shutil
import sys
import logging
import argparse
import time
import json
from torch.utils.tensorboard import SummaryWriter


def parse_args():

    parser = argparse.ArgumentParser(description='NTT')
    parser.add_argument('--gpu_id', type=str, default='2', 
                        help='gpu id')

    # data settings
    parser.add_argument('--data_path', type=str, 
                        help='data path')
    parser.add_argument('--res_path', type=str, 
                        help='result path of model, logs, outputs')
    parser.add_argument('--model_name', type=str, 
                        help='model name')
    
    parser.add_argument('--num_classes', type=int, default=2, 
                        help='number of classes')
    parser.add_argument('--n_pos_samples', type=int, default=-1,
                        help='the number of positive samples to use, default: -1 using all the positive samples')
    parser.add_argument('--n_neg_samples', type=int, default=-1,
                        help='the number of negative samples to use, default: -1 using all the negative samples')
    parser.add_argument('--shuffle', action='store_false',
                        help='if shuffling the dataset, default: True')
    parser.add_argument('--random_seed', type=int, default=1234, 
                        help='random seed')
    
    # train settings
    parser.add_argument('--lr', type=float, default=0.001, 
                        help='learning rate')
    parser.add_argument('--lr_center', type=float, default=0.0001, 
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, 
                        help='weight decay')
    parser.add_argument('--epochs', type=int, default=100, 
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='batch size')
    parser.add_argument('--patience', type=int, default=7, 
                        help='number of epochs for stopping if not updating')
    parser.add_argument('--log_interval', type=int, default=1, 
                        help='log interval')
    parser.add_argument('--use_node_aux_attr', action='store_true',
                        help='using node attributes, i.e., node names, default: False')
    parser.add_argument('--use_graph_aux_attr', action='store_true',
                        help='using graph attributes, i.e., time info, default: False')
    parser.add_argument('--use_unlabel', action='store_true',
                        help='using unlabeled graph data, default: False')
    parser.add_argument('--use_pretrain', action='store_true',
                        help='using pretrained model, default: False')
    parser.add_argument('--pretrain_model_name', type=str, default='',
                        help='pretrained model name')
    parser.add_argument('--reload_data', action='store_true',
                        help='reloading data, default: False')

    # model settings
    parser.add_argument('--h_dim', type=int, default=256, 
                        help='number of hidden dim')
    parser.add_argument('--graph_h_dim', type=int, default=256, 
                        help='number of graph hidden dim')
    parser.add_argument('--predict_func', type=str, default='SVM', 
                        help='the function to make prediction, NN | SVM')    
    parser.add_argument('--num_attn_layers', type=int, default=1, 
                        help='number of attentions, default: 1')
    parser.add_argument('--eta', type=float, default=0.1, 
                        help='scaler for labeled error, default: 0.1')
    
    # others
    parser.add_argument('--verbose', action='store_false', 
                        help='if logging the loss')

    args = parser.parse_args()
    return args


def verbose(params):

    if params['verbose']:
        for k, v in params.items():
            print(f'{k}: {v}')

    for k, v in params.items():
        logging.info(f'{k}: {v}')


def input_check(params):

    # check data existence
    if not os.path.exists(params['data_path']):
        print('The data path does not exist. SKIP.')
        sys.exit(1)

    # check model existence
    if not os.path.exists(params['res_path']):
        os.makedirs(params['res_path'])
        os.makedirs(params['model_path'])
        os.makedirs(params['log_path'])
        os.makedirs(params['img_path'])

    with open(params['output_json_path'], "w") as f:
        json.dump(params, f, indent=4)
            
    return params


    
def load_param_dict(args=None, mode='train'):
    
    param_dict = dict()
    param_dict['mode'] = mode
    
    if args is None:
        param_dict['gpu_id'] = 2
        param_dict['data_path'] = '/data/yijun/NTT/graph-data/08/1NTT19478_naive_itv_72'
        param_dict['res_path'] = '/data/yijun/NTT/model-results-1006'
        param_dict['model_name'] = 'TGConvNGATCluster2'
        
        param_dict['num_classes'] = 2
        param_dict['n_pos_samples'] = -1
        param_dict['n_neg_samples'] = -1
        param_dict['shuffle'] = True
        param_dict['random_seed'] = 1234
                   
        param_dict['lr'] = 0.001
        param_dict['lr_center'] = 0.0001
        param_dict['weight_decay'] = 0.0001
        param_dict['epochs'] = 200
        param_dict['batch_size'] = 32
        param_dict['patience'] = 7
        param_dict['use_node_aux_attr'] = False
        param_dict['use_graph_aux_attr'] = False
        param_dict['use_unlabel'] = False
        param_dict['use_pretrain'] = False
        param_dict['pretrain_model_name'] = ''
        param_dict['reload_data'] = False
        
    
        param_dict['h_dim'] = 256
        param_dict['graph_h_dim'] = 256
        param_dict['predict_func'] = 'SVM'
        param_dict['eta'] = 0.1
        
        param_dict['verbose'] = True
        
    else:
        param_dict['gpu_id'] = args.gpu_id
        param_dict['data_path'] = args.data_path
        param_dict['res_path'] = args.res_path
        param_dict['model_name'] = args.model_name
        
        param_dict['num_classes'] = args.num_classes
        param_dict['n_pos_samples'] = args.n_pos_samples
        param_dict['n_neg_samples'] = args.n_neg_samples
        param_dict['shuffle'] = args.shuffle
        param_dict['random_seed'] = args.random_seed
                   
        param_dict['lr'] = args.lr
        param_dict['lr_center'] = args.lr_center
        param_dict['weight_decay'] = args.weight_decay
        param_dict['epochs'] = args.epochs
        param_dict['batch_size'] = args.batch_size
        param_dict['patience'] = args.patience
        param_dict['use_node_aux_attr'] = args.use_node_aux_attr
        param_dict['use_graph_aux_attr'] = args.use_graph_aux_attr
        param_dict['use_unlabel'] = args.use_unlabel
        param_dict['use_pretrain'] = args.use_pretrain
        param_dict['pretrain_model_name'] = args.pretrain_model_name
        param_dict['reload_data'] = args.reload_data
        
        param_dict['h_dim'] = args.h_dim
        param_dict['graph_h_dim'] = args.graph_h_dim
        param_dict['predict_func'] = args.predict_func
        param_dict['eta'] = args.eta
        
        param_dict['verbose'] = args.verbose
    
    data_name = os.path.basename(param_dict['data_path'])
    model_name = '{}_{}'.format(param_dict['model_name'], data_name)

    param_dict['res_path'] = os.path.join(param_dict['res_path'], 
                                          model_name + '_' + str(int(time.time())))     
                
    param_dict['model_path'] = os.path.join(param_dict['res_path'], 'models')
    param_dict['log_path'] = os.path.join(param_dict['res_path'], 'logs')
    param_dict['img_path'] = os.path.join(param_dict['res_path'], 'imgs')
    param_dict['output_json_path'] = os.path.join(param_dict['res_path'], 'output.json')
    param_dict = input_check(param_dict)
    
    param_dict['log_file_path'] = os.path.join(param_dict['log_path'], model_name + f'_{mode}.log')
    logging.basicConfig(filename=param_dict['log_file_path'], level=logging.INFO, 
                        format='%(asctime)s %(message)s', datefmt='%m-%d %H:%M')
    
    verbose(param_dict)
    return param_dict