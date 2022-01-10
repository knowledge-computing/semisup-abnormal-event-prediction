import time
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

import sys
sys.path.append('../../')

from trainers.base_trainer import BaseTrainer


class TGConvNGATAETrainer(BaseTrainer):
    
    def __init__(
        self, 
        data, 
        model, 
        device,
        epochs: int = 50,
        lr: float = 0.01,
        lr_milestones: tuple = (),
        weight_decay: float = 0.01,
        res_path: str = '',
    ):
        
        super().__init__(data=data, 
                         model=model, 
                         device=device, 
                         epochs=epochs, 
                         lr=lr, 
                         weight_decay=weight_decay, 
                         res_path=res_path)
        
        self.lr_milestones = lr_milestones
        
    def train(self):

        mse_loss_func = nn.MSELoss(reduction='sum').to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), 
                                     lr=self.lr, 
                                     weight_decay=self.weight_decay)
    
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                                         milestones=self.lr_milestones, 
                                                         gamma=0.1)

        self.model.train()    
        for epoch in range(self.epochs):

            epoch_loss = 0.0
            num_samples = 0

            epoch_start_time = time.time()

            for i, (data, unsup_data) in enumerate(self.semi_loader):

                data = data.to(self.device)
                unsup_data = unsup_data.to(self.device)
                optimizer.zero_grad()

                _, node_rec = self.model(data)
                _, unsup_node_rec = self.model(unsup_data)
                loss = mse_loss_func(torch.cat((data.x, unsup_data.x), dim=0), 
                                     torch.cat((node_rec, unsup_node_rec), dim=0))   
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_samples += 1

            scheduler.step()
            if epoch in self.lr_milestones:
                print('LR scheduler: new learning rate is {}'.format(scheduler.get_last_lr()))
                logging.info('LR scheduler: new learning rate {}'.format(scheduler.get_last_lr()))

            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            loss_dict = {'MSE Loss': epoch_loss / num_samples}
            print(self.train_info(time=epoch_train_time, loss_dict=loss_dict))
            logging.info(self.train_info(time=epoch_train_time, loss_dict=loss_dict))
            self.cur_epoch += 1
            
            torch.save(self.model.state_dict(), os.path.join(self.res_path, 'models/model.pkl'))

            
    def test(self, data_loader):
    
        x, node_emb, node_rec = [], [], []

        self.model.eval()
        with torch.no_grad():

            for data in loader:

                data = data.to(self.device)
                emb, rec = self.model(data)
                x.append(data.x.cpu().numpy())
                node_emb.append(emb.cpu().numpy())
                node_rec.append(rec.cpu().numpy())

        x = np.concatenate(x, axis=0)
        node_emb = np.concatenate(node_emb, axis=0)
        node_rec = np.concatenate(node_rec, axis=0)

        mse_loss = mean_squared_error(x, node_res)
        print(f'MSE Loss: {mse_loss}')
        return node_emb