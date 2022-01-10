import os
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import sys
sys.path.append('../../')

from trainers.base_trainer import BaseTrainer


def target_distribution(batch: torch.Tensor) -> torch.Tensor:
    """
    Compute the target distribution p_ij, given the batch (q_ij), as in 3.1.3 Equation 3 of
    Xie/Girshick/Farhadi; this is used the KL-divergence loss function.
    :param batch: [batch size, number of clusters] Tensor of dtype float
    :return: [batch size, number of clusters] Tensor of dtype float
    """
    
    weight = (batch ** 2) / torch.sum(batch, 0)
    return (weight.t() / torch.sum(weight, 1)).t()
    
    
class ContrastiveCenterLoss(nn.Module):

    def __init__(self, num_classes, feat_dim, device, centers=None, lambda_c=1.0, weight=None):
        super(ContrastiveCenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device
        self.weight = weight
        
        if centers is None:
            self.centers = nn.Parameter(torch.randn(num_classes, feat_dim).to(device))
        else:
            self.centers = nn.Parameter(centers)

        self.lambda_c = lambda_c

    # may not work due to flowing gradient. change center calculation to exp moving avg may work.
    def forward(self, x, y):

        batch_size = x.size()[0]
        dist_centers = torch.sum((x.unsqueeze(1) - self.centers) ** 2, 2)
        dist_same = dist_centers.gather(1, y.unsqueeze(1))
        intra_dist = dist_same.sum()
        inter_dist = dist_centers.sum().sub(intra_dist)
        
        epsilon = 1e-6
        loss = (self.lambda_c / 2.0 / batch_size) * intra_dist / (inter_dist + epsilon) / 0.1
        return loss

    
class TGConvNGATClusterTrainer(BaseTrainer):
    
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
        eta: float = 0.1,
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
        self.centers = np.random.randn(self.num_classes, self.graph_h_dim)
        
        self.eta = eta

        
    def train(self):
        
        ce_loss_func = nn.CrossEntropyLoss(weight=self.label_weight.to(self.device))
        kl_loss_function = nn.KLDivLoss(reduction='sum')

#         centers = self.init_centers(data_loader=self.semi_loader)
        center_loss_func = ContrastiveCenterLoss(num_classes=self.num_classes, 
                                                 feat_dim=self.graph_h_dim, 
                                                 device=self.device, 
                                                 centers=None,
                                                 weight=self.label_weight.to(self.device)).to(self.device)
        
        optimizer = torch.optim.Adam(self.model.parameters(), 
                                     lr=self.lr, 
                                     weight_decay=self.weight_decay)
    
        center_optimizer = torch.optim.SGD(center_loss_func.parameters(), 
                                           lr=0.0001, momentum=0.9)

        self.model.train()    
        for epoch in range(self.epochs):
            
            epoch_loss = 0. 
            center_losses = 0. 
            kl_losses = 0.
            num_samples = 0

            epoch_start_time = time.time()
            
            for i, (data, unsup_data) in enumerate(self.semi_loader):

                loss = 0.
                data = data.to(self.device)
                unsup_data = unsup_data.to(self.device)
                optimizer.zero_grad()
                center_optimizer.zero_grad()  

                node_emb, graph_emb, out = self.model(data)
                unsup_node_emb, unsup_graph_emb, unsup_out = self.model(unsup_data)
                emb = torch.cat((graph_emb, unsup_graph_emb), dim=0)
                
                def _assignment(g_emb):
                    dist_centers = torch.sum((g_emb.unsqueeze(1) - center_loss_func.centers) ** 2, 2)
                    norm_dis = 1. / (1. + dist_centers)
                    ass_prob = norm_dis / torch.sum(norm_dis, dim=1, keepdim=True)
                    return ass_prob, torch.argmin(dist_centers, dim=1)
                
                center_loss = center_loss_func(graph_emb, data.y)
                loss += center_loss
                
                assignment, _ = _assignment(emb)
                target = target_distribution(assignment).detach()  # num_sample * num_clusters
                target[:graph_emb.shape[0]] = F.one_hot(data.y, num_classes=2).float()
                kl_loss = kl_loss_function(assignment.log(), target) / assignment.shape[0]
                loss += kl_loss * self.eta
                
                loss.backward()
                optimizer.step()
                center_optimizer.step()

                epoch_loss += loss.item()
                center_losses += center_loss.item()
                kl_losses += kl_loss.item()
                num_samples += 1

            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            loss_dict = {'Train Loss': epoch_loss / num_samples,
                         'Center Loss': center_losses / num_samples,
                         'KL Loss': kl_losses / num_samples,
                        }
            print(center_loss_func.centers.detach().cpu().numpy()[:, :5])
            print(self.train_info(time=epoch_train_time, loss_dict=loss_dict))
            logging.info(self.train_info(time=epoch_train_time, loss_dict=loss_dict))
            self.cur_epoch += 1
            
            """ validate """            
            if epoch % self.log_interval == 0:
                self.validate(center=center_loss_func.centers.detach().cpu().numpy())

            if self.best_val_count >= self.patience:
                self.centers = center_loss_func.centers.detach().cpu().numpy()
                break
                
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
    
        # choose the center with closer distance 
#         extended_graph_emb = np.repeat(np.expand_dims(graph_emb, axis=0), self.centers.shape[0], axis=0)
#         extended_centers = np.repeat(np.expand_dims(self.centers, axis=1), graph_emb.shape[0], axis=1)
#         mse = np.sum((extended_graph_emb - extended_centers) ** 2, axis=-1)
#         pred = np.argmin(mse, axis=0)       
        pred = np.argmax(np.concatenate(pred, axis=0), axis=1)
        y = np.concatenate(y, axis=0)
        return graph_emb, pred, y
