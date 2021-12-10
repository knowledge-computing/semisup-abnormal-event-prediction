import os
import time
import pickle
import numpy as np
import logging
import torch
from abc import ABC, abstractmethod

from semi_supervised_AD.models.others import svm_classify, svm_predict

from semi_supervised_AD.utils.plot_utils import draw_train_test_plot
from semi_supervised_AD.utils.plot_utils import draw_train_test_center_plot
from semi_supervised_AD.utils.plot_utils import draw_individual_plots
from semi_supervised_AD.utils.evaluate import evaluate


class BaseTrainer(ABC):
    """ trainer base class """

    def __init__(self, data, model, device,
                 epochs: int = 50,
                 lr: float = 0.01,
                 weight_decay: float = 0.01,
                 res_path: str = '',
                 verbose: bool = True):
        
        super().__init__()
        
        # data
        self.semi_loader = data.semi_loader
        self.train_loader = data.train_loader
        self.val_loader = data.val_loader
        self.test_loader = data.test_loader
        self.label_weight = data.label_weight

        # model        
        self.model = model
        self.clf = None
        self.device = device
        
        # train
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.res_path = res_path
        self.verbose = verbose

        self.cur_epoch = 0
        self.best_val_loss = None
        self.best_val_count = 0
        
    @abstractmethod
    def train(self):
        """
        Implement train method that trains the given network using the train_set of dataset.
        :return: Trained net
        """
        pass

    @abstractmethod
    def test(self):
        """
        Implement test method that evaluates the test_set of dataset on the given network.
        """
        pass
    
    def validate(self, center=None):
        
        train_emb, train_p, train_y = self.test(self.train_loader)
        val_emb, val_p, val_y = self.test(self.val_loader)
        test_emb, test_p, test_y = self.test(self.test_loader)

        if self.predict_func == 'SVM':
            self.clf = svm_classify(train_emb, train_y, self.label_weight.detach().cpu().numpy())
            train_p = svm_predict(train_emb, self.clf)
            val_p = svm_predict(val_emb, self.clf)
            test_p = svm_predict(test_emb, self.clf)
            
        val_loss = evaluate(val_y, val_p)
        test_loss = evaluate(test_y, test_p)
        
        logging.info(self.test_info(name='Val', loss=val_loss))
        logging.info(self.test_info(name='Test', loss=test_loss))
        if self.verbose:
            print(self.test_info(name='Val', loss=val_loss))
            print(self.test_info(name='Test', loss=test_loss))
    
        if self.best_val_loss is None or val_loss[-1] > self.best_val_loss:
            self.save_models(val_loss)
            self.plotting(x_dict={'train_x': train_emb, 'val_x': val_emb, 'test_x': test_emb},
                          y_dict={'train_y': train_y, 'val_y': val_y, 'test_y': test_y},
                          p_dict={'train_p': train_p, 'val_p': val_p, 'test_p': test_p},
                          center=center,)
        else:
            self.best_val_count += 1
            
    def save_models(self, val_loss):
        
        self.best_val_loss, self.best_val_count = val_loss[-1], 0
        torch.save(self.model.state_dict(), 
                   os.path.join(self.res_path, 'models/model.pkl'))
            
        if self.clf is not None:
            pickle.dump(self.clf, open(
            os.path.join(self.res_path, 'models/svm.pkl'), 'wb'))
            
        logging.info('>>> Models have been saved.')
        print('>>> Models have been saved.')
                    
    def plotting(self, x_dict, y_dict, p_dict, center=None):
        
        if self.predict_func == 'SVM':
            plot_dict = {'train': (self.clf.steps[0][1].transform(x_dict['train_x']), 
                                   y_dict['train_y'], p_dict['train_p']),
                         'val': (self.clf.steps[0][1].transform(x_dict['val_x']), 
                                 y_dict['val_y'], p_dict['val_p']),
                         'test': (self.clf.steps[0][1].transform(x_dict['test_x']), 
                                 y_dict['test_y'], p_dict['test_p'])}
        else:
            plot_dict = {'train': (x_dict['train_x'], y_dict['train_y'], p_dict['train_p']),
                         'val': (x_dict['val_x'], y_dict['val_y'], p_dict['val_p']),
                         'test': (x_dict['test_x'], y_dict['test_y'], p_dict['test_p'])}
            
        plot_dict['epoch'] = self.cur_epoch
        if center is None:
            draw_train_test_plot(plot_dict, self.res_path)
        else:
            plot_dict['center'] = center
            draw_train_test_center_plot(plot_dict, self.res_path)
            
        draw_individual_plots(plot_dict, self.res_path)

    def test_info(self, name, loss):
        return '{} ACC | p0:{:.3f}, r0:{:.3f}, p1:{:.3f}, r1:{:.3f}, f1:{:.3f}.' \
               .format(name, *loss)
    
    def train_info(self, time, loss_dict):
        s = 'Epoch: {} / {} | Train Time: {:.3f} | '.format(self.cur_epoch, self.epochs, time)
        for loss_name, loss in loss_dict.items():
            s += '{}: {:.6f}; '.format(loss_name, loss)
        return s


         