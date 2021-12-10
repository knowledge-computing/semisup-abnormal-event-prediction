import os
import shutil
import sys
import logging
import argparse
import time
from torch.utils.tensorboard import SummaryWriter


def arg_parse():

    parser = argparse.ArgumentParser(description='NTT')
    parser.add_argument('--device', type=str, default='2', 
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
    parser.add_argument('--use_tb', action='store_true', 
                        help='default: False')
    parser.add_argument('--tb_path', type=str, default='', 
                        help='tensorboard path')
    parser.add_argument('--verbose', action='store_false', 
                        help='if logging the loss')

    args = parser.parse_args()
    input_check(args)
    return args


def verbose(args):

    if args.verbose:
        for arg in vars(args):
            print(f'{arg}: {getattr(args, arg)}')

    for arg in vars(args):
        logging.info(f'{arg}: {getattr(args, arg)}')


def input_check(args):

    # check data existence
    if not os.path.exists(args.data_path):
        print('The data path does not exist. SKIP.')
        sys.exit(1)

    # check model existence
    if not os.path.exists(args.res_path):
        os.mkdir(args.res_path)
    if os.path.exists(os.path.join(args.res_path, 'img')):
        shutil.rmtree(os.path.join(args.res_path, 'img'))
    if os.path.exists(os.path.join(args.res_path, 'models')):
        shutil.rmtree(os.path.join(args.res_path, 'models'))
    
    os.mkdir(os.path.join(args.res_path, 'img'))
    os.mkdir(os.path.join(args.res_path, 'models'))
       
    # check log path
    if args.use_tb and not os.path.exists(args.tb_path):
        print('The tensorboard path does not exist. SKIP.')
        sys.exit(1)

        
def initialize_tb(args):
    ''' initialize tensor board '''
    
    if args.use_tb:
        tb_path = os.path.join(args.tb_path, args.model_name)
        tb_writer = SummaryWriter(tb_path)
    else:
        tb_writer = None
    return tb_writer    