import os
import shutil
import sys
import logging
import argparse
import time
from torch.utils.tensorboard import SummaryWriter


def arg_parse():

    parser = argparse.ArgumentParser(description='NTT')
    parser.add_argument('--gpu_id', type=str, default='2', 
                        help='gpu id')

    # data settings
    parser.add_argument('--res_path', type=str, 
                        help='result path of model, logs, outputs')
    parser.add_argument('--model_name', type=str, 
                        help='model name') 
    args = parser.parse_args()
    input_check(args)
    return args


def input_check(params):

    # check data existence
    if not os.path.exists(params['data_path']):
        print('The data path does not exist. SKIP.')
        sys.exit(1)

    # check model existence
    if not os.path.exists(params['res_path']):
        print('The result path does not exist. SKIP.')
        sys.exit(1)

        
def load_param_dict(args=None, mode='test'):
    
    param_dict = dict()
    param_dict['mode'] = mode

    if args is None:
        param_dict['model_name'] = 'test'
        param_dict['res_path'] = './results'
        param_dict['gpu_id'] = 1
    else:
        param_dict['model_name'] = args.model_name
        param_dict['res_path'] = args.res_path
        param_dict['gpu_id'] = args.gpu_id
        
    param_dict['res_path'] = os.path.join(param_dict['res_path'], param_dict['model_name'])            
    param_dict['model_path'] = os.path.join(param_dict['res_path'], 'models')
    param_dict['log_path'] = os.path.join(param_dict['res_path'], 'logs')
    param_dict['img_path'] = os.path.join(param_dict['res_path'], 'imgs')
    
    output_json_path = os.path.join(param_dict['res_path'], 'output.json')
    with open(output_json_path, "r") as f:
        output_json = json.load(f)
    
    for k, v in output_json.items():
        if k not in param_dict.keys():
            param_dict[k] = v
    print(param_dict)
    param_dict = input_check(param_dict)
    return param_dict