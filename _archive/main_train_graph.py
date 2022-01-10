import os
import logging
import pickle
import datetime

import sys 
sys.path.append("..") 

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import DataLoader, Dataset

from anomaly_detection.data_loader.gen_train_val_test import compute_label_weight, split_train_val_test, ConcatDataset
from anomaly_detection.data_loader.ntt_flatten_loader import NTTDataset
from anomaly_detection.scripts.test_graph import test_and_evaluate, test, svm_classify_and_predict, evaluate
from anomaly_detection.scripts.loss import CenterLoss, ContrastiveCenterLoss, FocalLoss
from anomaly_detection.scripts.helper import initialize_model, run_train_func
from anomaly_detection.utils.plot_utils import draw_train_test_plot, draw_train_test_plot_1, draw_individual_plots
from anomaly_detection.utils.train_options import arg_parse, verbose

    
if __name__ == '__main__':

    args = arg_parse()
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    
    ''' logging start '''
    log_file = os.path.join(args.res_path, f'{args.model_name}.log')
    logging.basicConfig(filename=log_file, level=logging.INFO, 
                        format='%(asctime)s %(message)s', datefmt='%m-%d %H:%M')

    ''' tensor board '''
    if args.use_tb:
        tb_path = os.path.join(args.tb_path, args.model_name)
        tb_writer = SummaryWriter(tb_path)
    else:
        tb_writer = None
        
    verbose(args)
    
    ''' loading data '''
    dataset = NTTDataset(root=args.data_path)
    train_dataset, val_dataset, test_dataset, unsup_dataset = split_train_val_test(dataset, args.num_classes, 
                                                                                   shuffle=args.shuffle, 
                                                                                   use_unlabel=args.use_unlabel, 
                                                                                   random_seed=args.random_seed)
    
    if args.use_unlabel:
        train_loader = DataLoader(ConcatDataset(train_dataset, unsup_dataset), 
                                  batch_size=args.batch_size, shuffle=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    train_loader_unshuffle = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    ''' compute the weight for labels '''
    label_weight = compute_label_weight(train_dataset, args.num_classes)
    print('weighted labels: {}'.format(label_weight))
    logging.info('weighted labels: {}'.format(label_weight))
    label_weight = torch.tensor(label_weight, dtype=torch.float).to(device)
    
    ''' select model '''
    model = initialize_model(dataset, args).to(device)
    
    center_loss_func = ContrastiveCenterLoss(num_classes=args.num_classes, 
                                             feat_dim=args.graph_h_dim, device=device)
    loss_func = {
        'nll_loss_func': torch.nn.NLLLoss(weight=None),
        'weighted_nll_loss_func': torch.nn.NLLLoss(weight=label_weight),
        'ce_loss_func': torch.nn.CrossEntropyLoss(weight=None),
        'weighted_ce_loss_func': torch.nn.CrossEntropyLoss(weight=label_weight),
        'center_loss_func': center_loss_func,
        'bce_loss_func': torch.nn.BCEWithLogitsLoss(weight=None), #, reduction='sum'),
        'weighted_bce_loss_func': torch.nn.BCEWithLogitsLoss(weight=label_weight, reduction='sum'),
        'mse_loss_func': torch.nn.MSELoss(),
        'mse_loss_sum_func': torch.nn.MSELoss(reduction='sum'),
    }
    
    optimizer = {
        'base_optim': torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay),
        'center_optim': torch.optim.SGD(center_loss_func.parameters(), lr=args.lr_center),
    }

    def loss_info(name, ep, lo):
        return 'Epoch: {} | {} p0:{:.3f}, p1:{:.3f}, r0:{:.3f}, r1:{:.3f}, f1:{:.3f}.' \
               .format(ep, name, *lo)

#     val_loss, test_loss = test_and_evaluate(model, train_loader_unshuffle, 
#                                             val_loader, test_loader, 
#                                             label_weight, device)
    
#     logging.info(loss_info(name='Val', ep=0, lo=val_loss))
#     logging.info(loss_info(name='Test', ep=0, lo=test_loss))
#     if verbose:
#         print(loss_info(name='Val', ep=0, lo=val_loss))
#         print(loss_info(name='Test', ep=0, lo=test_loss))

    best_val_loss, best_val_count = None, 0
    for epoch in range(1, args.epochs + 1):
        
        """ train """
        
        cur_time = datetime.datetime.now()
        loss = run_train_func(dataset, train_loader, 
                              model, optimizer, loss_func, 
                              device, args)
        time_delta = (datetime.datetime.now() - cur_time)
        print(f'=== Epoch {epoch} Training Time: {time_delta} ===')

        if args.use_tb:
            tb_writer.add_scalar('Loss/train_loss', float(loss), epoch)

        """ test """            
        if epoch % args.log_interval == 0:

            _, train_enc, train_y, train_p, _ = test(model, train_loader_unshuffle, device)
            _, val_enc, val_y, val_p, _ = test(model, val_loader, device)
            _, test_enc, test_y, test_p, _ = test(model, test_loader, device)

            clf = None
            if args.predict_func == 'SVM':
                clf, p_dict = svm_classify_and_predict(train_enc, train_y, val_enc, test_enc, label_weight)
                val_p, test_p = p_dict['val_p'], p_dict['test_p']

            val_loss = evaluate(val_y, val_p)
            test_loss = evaluate(test_y, test_p)
            logging.info(loss_info(name='Val', ep=epoch, lo=val_loss))
            logging.info(loss_info(name='Test', ep=epoch, lo=test_loss))
            if verbose:
                print(loss_info(name='Val', ep=epoch, lo=val_loss))
                print(loss_info(name='Test', ep=epoch, lo=test_loss))
            
            if best_val_loss is None or val_loss[-1] > best_val_loss:
                best_val_loss, best_val_count = val_loss[-1], 0
                torch.save(model.state_dict(), os.path.join(args.res_path, 'models/model.pkl'))

                if args.predict_func == 'SVM':
                    pickle.dump(clf, open(os.path.join(args.res_path, 'models/svm.pkl'), 'wb'))
                    plot_dict = {'train': (clf.steps[0][1].transform(train_enc), train_y, train_p, epoch),
                                 'val': (clf.steps[0][1].transform(val_enc), val_y, val_p, epoch),
                                 'test': (clf.steps[0][1].transform(test_enc), test_y, test_p, epoch)}
                else:
                    plot_dict = {'train': (train_enc, train_y, train_p, epoch),
                                 'val': (val_enc, val_y, val_p, epoch),
                                 'test': (test_enc, test_y, test_p, epoch)}
                draw_train_test_plot_1(plot_dict, args.res_path)
                draw_individual_plots(plot_dict, args.res_path)

                logging.info('>>> Models have been saved.')
                print('>>> Models have been saved.')
            else:
                best_val_count += 1

        if best_val_count >= args.patience:
            break


