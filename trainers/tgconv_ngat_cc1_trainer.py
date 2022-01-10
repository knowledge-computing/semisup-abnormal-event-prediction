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


class ContrastiveCenterLoss(nn.Module):

    def __init__(self, num_classes, feat_dim, device, centers=None, lambda_c=1.0):
        super(ContrastiveCenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device
#         print(centers[0:1].shape)
        
        if centers is None:
            self.centers = nn.Parameter(torch.randn(1, feat_dim).to(device))
        else:
            self.centers = nn.Parameter(centers[0:1])

#         self.r = nn.Parameter(torch.randn(1).to(device))
        self.lambda_c = lambda_c

    # may not work due to flowing gradient. change center calculation to exp moving avg may work.
    def forward(self, x, y):

        batch_size = x.size()[0]
        expanded_centers = self.centers.expand(batch_size, -1)
        dist_centers = (x - expanded_centers).pow(2).sum(dim=-1)
        intra_dist = dist_centers[y == 1].sum() / torch.sum(y == 1)
        if torch.sum(y == 1) > 0:
            inter_dist = dist_centers[y == 0].sum() / torch.sum(y == 0)
            loss = self.lambda_c * intra_dist / (inter_dist + 1e-6)
        else:
            inter_dist = dist_centers[y == 0].sum() / torch.sum(y == 0)
            loss = self.lambda_c / (inter_dist + 1e-6)
        return loss

    
class TGConvNGATCC1Trainer(BaseTrainer):
    
    def __init__(
        self, 
        data, 
        model, 
        device, 
        epochs: int = 50,
        lr: float = 0.01,
        lr_center: float = 0.001,
        weight_decay: float = 0.01,
        patience: int = 7,
        predict_func: str = 'SVM',
        res_path: str = '',
    ):
        
        super().__init__(data=data, 
                         model=model, 
                         device=device, 
                         epochs=epochs, 
                         lr=lr, 
                         weight_decay=weight_decay, 
                         res_path=res_path,)
         
        self.num_classes = model.out_dim
        self.graph_h_dim = model.graph_h_dim
        
        self.lr_center = lr_center
        self.patience = patience
        self.predict_func = predict_func
        
        self.alpha = 10
        self.eps = 1e-6
        
    def train(self):
        
        centers = None  # self.init_centers(data_loader=self.semi_loader)
        center_loss_func = ContrastiveCenterLoss(num_classes=self.num_classes, 
                                                 feat_dim=self.graph_h_dim, 
                                                 device=self.device,
                                                 centers=centers)
        
        optimizer = torch.optim.Adam(self.model.parameters(), 
                                     lr=self.lr, 
                                     weight_decay=self.weight_decay)
    
        center_optimizer = torch.optim.SGD(center_loss_func.parameters(), 
                                           lr=self.lr_center)

        self.model.train()    
        for epoch in range(self.epochs):
            
            epoch_loss, center_losses = 0.0, 0.0
            num_samples = 0

            epoch_start_time = time.time()
            
            for i, (data, unsup_data) in enumerate(self.semi_loader):

                data = data.to(self.device)
                unsup_data = unsup_data.to(self.device)
                optimizer.zero_grad()
                center_optimizer.zero_grad()  

                node_emb, graph_emb, out = self.model(data)
                unsup_node_emb, unsup_graph_emb, unsup_out = self.model(unsup_data)
                emb = torch.cat((graph_emb, unsup_graph_emb), dim=0)

                center_loss = center_loss_func(graph_emb, data.y)
                loss = center_loss
                
                loss.backward()
                optimizer.step()
                
                # multiple (1./alpha) in order to remove the effect of alpha on updating centers
                for param in center_loss_func.parameters():
                    param.grad.data *= (1. / self.alpha)
                center_optimizer.step()

                epoch_loss += loss.item()
                center_losses += center_loss.item()
                num_samples += 1

            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            loss_dict = {'Train Loss': epoch_loss / num_samples,
                         'Center Loss': center_losses / num_samples}
            print(self.train_info(time=epoch_train_time, loss_dict=loss_dict))
            logging.info(self.train_info(time=epoch_train_time, loss_dict=loss_dict))
            self.cur_epoch += 1
            
            """ validate """            
            if epoch % self.log_interval == 0:
                self.validate(center=center_loss_func.centers.detach().cpu().numpy())

            if self.best_val_count >= self.patience:
                break
                
            if self.best_val_count == self.patience // 2: 
                optimizer.param_groups[0]['lr'] *= 0.1
                print(f'Val loss does not decrease for {self.best_val_count} epochs.')
                print('New learning rate: {}.'.format(optimizer.param_groups[0]['lr']))

    def init_centers(self, data_loader):
        
        n_neg_samples, n_pos_samples = 0, 0
        centers = torch.zeros((self.num_classes, self.graph_h_dim), device=self.device)
        self.model.eval()
        with torch.no_grad():
            
            for i, (data, _) in enumerate(data_loader):

                data = data.to(self.device)
                _, graph_emb, _ = self.model(data)
                n_neg_samples += torch.sum(data.y == 0)
                n_pos_samples += torch.sum(data.y == 1)
                centers[0] += torch.sum(graph_emb[data.y == 0], dim=0)
                centers[1] += torch.sum(graph_emb[data.y == 1], dim=0)
        
        centers[0] = centers[0] / n_neg_samples
        centers[1] = centers[1] / n_pos_samples
        return centers
    
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
