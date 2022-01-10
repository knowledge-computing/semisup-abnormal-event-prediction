import os
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('../../')

from trainers.base_trainer import BaseTrainer


class TGConvNGATTrainer(BaseTrainer):
    
    def __init__(
        self, 
        data, 
        model, 
        device, 
        eta,
        epochs: int = 50,
        lr: float = 0.01,
        weight_decay: float = 0.01,
        patience: int = 7,
        predict_func: str = 'SVM',
        res_path: str = '',):
        
        super().__init__(data=data, 
                         model=model, 
                         device=device, 
                         epochs=epochs, 
                         lr=lr, 
                         weight_decay=weight_decay, 
                         res_path=res_path,)
         
        self.patience = patience
        self.predict_func = predict_func
        
        self.c = torch.zeros(model.graph_h_dim, device=device)
        self.eta = eta
        
    def train(self):
        
        ce_loss_func = torch.nn.CrossEntropyLoss(weight=self.label_weight.to(self.device))
        optimizer = torch.optim.Adam(self.model.parameters(), 
                                     lr=self.lr, 
                                     weight_decay=self.weight_decay)
        
        self.init_center_c(data_loader=self.semi_loader)
        print(f'Center c has been initialized: {self.c.shape}')
    
        self.model.train()    
        for epoch in range(self.epochs):
            
            epoch_losses = 0.0
            ce_losses = 0.0
            sphere_losses = 0.0
            num_samples = 0

            epoch_start_time = time.time()
            
            for i, (data, unsup_data) in enumerate(self.semi_loader):

                data = data.to(self.device)
                unsup_data = unsup_data.to(self.device)
                optimizer.zero_grad()

                node_emb, graph_emb, out = self.model(data)
                unsup_node_emb, unsup_graph_emb, unsup_out = self.model(unsup_data)
                emb = torch.cat((graph_emb, unsup_graph_emb), dim=0)

                ce_loss = ce_loss_func(out, data.y)

                # modify the labels to -1: abnormal, 1: normal, 0: unlabel
                y = torch.cat((-data.y * 2 + 1, unsup_data.y + 1), dim=0)

                dist = torch.sum((emb - self.c) ** 2, dim=1)
                sphere_loss = torch.where(y == 0, dist, self.eta * ((dist + self.eps) ** y.float()))
                sphere_loss = torch.mean(sphere_loss)
                
                loss = ce_loss + sphere_loss
                loss.backward()
                optimizer.step()

                ce_losses += ce_loss
                sphere_losses += sphere_loss
                epoch_losses += loss.item()
                num_samples += 1

            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            loss_dict = {'Train Loss': epoch_losses / num_samples,
                         'CE Loss': ce_losses / num_samples,
                         'SPHERE Loss': sphere_losses / num_samples}
            print(self.train_info(time=epoch_train_time, loss_dict=loss_dict))
            logging.info(self.train_info(time=epoch_train_time, loss_dict=loss_dict))
            self.cur_epoch += 1
            
            """ validate """            
            if epoch % self.log_interval == 0:
                self.validate(center=self.c.detach().cpu().numpy().reshape(1, -1))

            if self.best_val_count >= self.patience:
                break
                
            if self.best_val_count == self.patience // 2: 
                optimizer.param_groups[0]['lr'] *= 0.1
                print(f'Val loss does not decrease for {self.best_val_count} epochs.')
                print('New learning rate: {}.'.format(optimizer.param_groups[0]['lr']))
                
    def init_center_c(self, data_loader):

        n_samples = 0
        
        self.model.eval()
        with torch.no_grad():
            
            for i, (data, unsup_data) in enumerate(data_loader):

                data = data.to(self.device)
                unsup_data = unsup_data.to(self.device)

                _, graph_emb, _ = self.model(data)
                _, unsup_graph_emb, _ = self.model(unsup_data)
                emb = torch.cat((graph_emb, unsup_graph_emb), dim=0)

                n_samples += emb.shape[0]
                self.c += torch.sum(emb, dim=0)

        self.c /= n_samples

        # If c_i is too close to 0, set to +-eps. 
        # Reason: a zero unit can be trivially matched with zero weights.
        self.c[(abs(self.c) < self.eps) & (self.c < 0)] = -self.eps
        self.c[(abs(self.c) < self.eps) & (self.c > 0)] = self.eps
        
    def test(self, data_loader):

        graph_emb, pred, y = [], [], []

        self.model.eval()
        with torch.no_grad():
            
            for data in data_loader:
                
                data = data.to(self.device)
                _, emb, p = self.model(data)
                p = F.log_softmax(p, dim=1)
                graph_emb.append(emb.cpu().numpy())
                pred.append(p.cpu().numpy())
                y.append(data.y.cpu().numpy())

        graph_emb = np.concatenate(graph_emb, axis=0)
        pred = np.argmax(np.concatenate(pred, axis=0), axis=1)
        y = np.concatenate(y, axis=0)
        return graph_emb, pred, y
