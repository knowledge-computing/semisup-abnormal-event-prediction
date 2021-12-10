import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import DataLoader, Batch
from anomaly_detection.scripts.loss import global_global_cdist_batch_loss, cvae_loss


def one_hot_encoding(labels): 
    targets = torch.zeros(labels.size(0), 2)
    for i, label in enumerate(labels):
        targets[i, label] = 1
    return targets


def special_one_hot_encoding(labels, num_classes=7): 
    targets = torch.zeros(labels.size(0), num_classes)
    for i, label in enumerate(labels):
        targets[i, 1] = 1
        targets[i, label] = 1
        if label < 0: continue
        elif label <  num_classes - 1:
            targets[i, label + 1] = 1
        else:
            targets[i, 0] = 1
    return targets


def verbose(loss_dict):
    s = ''
    for loss_name, loss in loss_dict.items():
        s += '{}: {:.3f}; '.format(loss_name, loss)
    print(s)
    logging.info(s)
    

def train_tgconv_ngat_ae(data_loader, model, device)
    
    
    



def train(data_loader, model, optimizer, loss_func, device):

    ce_loss_func = loss_func['weighted_ce_loss_func']
    base_optim = optimizer['base_optim']
    
    model.train()
    loss_all, ce_losses = 0., 0.
    
    for data in data_loader:

        data = data.to(device)
        _, g_enc, _, g_out = de, mu, log_var, z(data)
        ce_loss = ce_loss_func(g_out, data.y)

        base_optim.zero_grad()
        loss.backward()
        base_optim.step()

        ce_losses += ce_loss.item()
        loss_all += loss.item()

    loss_dict = {
        'Total Loss': loss_all,
        'CE Loss': ce_losses,
    }
    verbose(loss_dict)
    return loss_all


def train_center_loss(data_loader, model, optimizer, loss_func, device):

    ce_loss_func = loss_func['weighted_ce_loss_func']
    center_loss_func = loss_func['center_loss_func']
    base_optim = optimizer['base_optim']
    center_optim = optimizer['center_optim']
    
    alpha = 5
    model.train()
    loss_all, center_losses, ce_losses = 0., 0., 0.
    
    for data in data_loader:

        data = data.to(device)
        _, g_enc, _, g_out = model(data)
        ce_loss = ce_loss_func(g_out, data.y)
        center_loss = center_loss_func(g_enc, data.y)
        loss = ce_loss + center_loss * alpha
        
        base_optim.zero_grad()
        center_optim.zero_grad()
        loss.backward()
        base_optim.step()
        
        # multiple (1./alpha) in order to remove the effect of alpha on updating centers
        for param in center_loss_func.parameters():
            param.grad.data *= (1. / alpha)
        center_optim.step()
        
        ce_losses += ce_loss.item()
        center_losses += center_loss.item()
        loss_all += loss.item()

    loss_dict = {
        'Total Loss': loss_all,
        'CE Loss': ce_losses,
        'Center Loss': center_losses,
    }
    verbose(loss_dict)
    return loss_all


def train_cvae(data_loader, model, optimizer, loss_func, device):

    ce_loss_func = loss_func['weighted_ce_loss_func']
    base_optim = optimizer['base_optim']
    m = model['gatconv']
    cvae = model['cvae']

    model.train()
    loss_all, bce_losses, kld_losses, ce_losses = 0., 0., 0., 0.
    
    for data in data_loader:

        data = data.to(device)
        c = one_hot_encoding(data.y).to(device)
        _, g_enc, g_out, de, mu, log_var, z = model(data, c)
        ce_loss = ce_loss_func(g_out, data.y)
        bce, kld, loss = cvae_loss(g_enc, de, mu, log_var)     
        loss = loss / 1000 + ce_loss
        
        base_optim.zero_grad()
        loss.backward()
        base_optim.step()

        bce_losses += bce.item()
        kld_losses += kld.item()
#         ce_losses += ce_loss.item()
        loss_all += loss.item()

    loss_dict = {
        'Total Loss': loss_all,
        'Reconstruct Loss': bce_losses,
        'KL Loss': kld_losses,
        'CE Loss': ce_losses
    }
    verbose(loss_dict)
    return loss_all



def train_cvae_center_loss(data_loader, model, optimizer, loss_func, device):

    ce_loss_func = loss_func['weighted_ce_loss_func']
    center_loss_func = loss_func['center_loss_func']
    bce_loss_func = loss_func['bce_loss_func']  # binary_cross_entropy_with_logits
    base_optim = optimizer['base_optim']
    center_optim = optimizer['center_optim']
    
    alpha = 10
    model.train()
    loss_all, bce_losses, kld_losses, ce_losses, center_losses = 0., 0., 0., 0., 0.

    for data in data_loader:

        data = data.to(device)
        c = one_hot_encoding(data.y).to(device)
        _, g_enc, g_out, de, mu, log_var, z = model(data, c)
        ce_loss = ce_loss_func(g_out, data.y)
        bce, kld, loss = cvae_loss(g_enc, de, mu, log_var)   
        center_loss = center_loss_func(g_enc, data.y)

        loss = loss / 1000
#         loss += ce_loss
        loss += center_loss * 10

        base_optim.zero_grad()
        center_optim.zero_grad()
        loss.backward()
        base_optim.step()
        
        # multiple (1./alpha) in order to remove the effect of alpha on updating centers
        for param in center_loss_func.parameters():
            param.grad.data *= (1. / alpha)
        center_optim.step()
                
        bce_losses += bce.item()
        kld_losses += kld.item()
        ce_losses += ce_loss.item()
        center_losses += center_loss.item()
        loss_all += loss.item()

    loss_dict = {
        'Total Loss': loss_all,
        'Reconstruct Loss': bce_losses,
        'KL Loss': kld_losses,
        'CE Loss': ce_losses,
        'Center Loss': center_losses,
    }
    verbose(loss_dict)
    return loss_all


def train_semi_cave_center_loss(data_loader, model, optimizer, loss_func, device):

    ce_loss_func = loss_func['weighted_ce_loss_func']
    center_loss_func = loss_func['center_loss_func']  
    bce_loss_func = loss_func['bce_loss_func']  # binary_cross_entropy_with_logits
    base_optim = optimizer['base_optim']
    center_optim = optimizer['center_optim']
    
    alpha = 10
    model.train()
    loss_all, bce_losses, kld_losses, ce_losses, center_losses = 0., 0., 0., 0., 0.
    
    for i, (data, unsup_data) in enumerate(data_loader):
        
        data = data.to(device)        
        unsup_data = unsup_data.to(device)
        
        c = one_hot_encoding(data.y).to(device)
        _, g_enc, g_out, de, mu, log_var, z = model(data, c)
        ce_loss = ce_loss_func(g_out, data.y)
        center_loss = center_loss_func(g_enc, data.y)
         
        _, unsup_g_enc = model.encoder(unsup_data)
        num_classes = center_loss_func.centers.size(0)
        num_unsup = unsup_g_enc.size(0)
        unsup_g_enc_ = unsup_g_enc.unsqueeze(0).repeat(num_classes, 1, 1)
        centers_ = center_loss_func.centers.detach().unsqueeze(1).repeat(1, num_unsup, 1)
        mse = torch.sum((unsup_g_enc_ - centers_) ** 2, dim=-1)
        unsup_y = torch.argmin(mse, dim=0)

        unsup_c = one_hot_encoding(unsup_y).to(device)
        unsup_de, unsup_mu, unsup_log_var, _ = model.cvae(unsup_g_enc, unsup_c)

        bce, kld, loss = cvae_loss(torch.cat((g_enc, unsup_g_enc), dim=0), 
                                   torch.cat((de, unsup_de), dim=0), 
                                   torch.cat((mu, unsup_mu), dim=0), 
                                   torch.cat((log_var, unsup_log_var), dim=0))   

        loss = loss / 1000 / 2
#         loss += ce_loss
        loss += center_loss * 10

        base_optim.zero_grad()
        center_optim.zero_grad()           
        loss.backward()
        base_optim.step()
        
        # multiple (1./alpha) in order to remove the effect of alpha on updating centers
        for param in center_loss_func.parameters():
            param.grad.data *= (1. / alpha)
        center_optim.step()
        
        ce_losses += ce_loss.item()
        center_losses += center_loss.item()
        bce_losses += bce.item()
        kld_losses += kld.item()
        loss_all += loss.item()

    loss_dict = {
        'Total Loss': loss_all,
        'Reconstruct Loss': bce_losses,
        'KL Loss': kld_losses,
        'CE Loss': ce_losses,
        'Center Loss': center_losses,
    }
    verbose(loss_dict)
    return loss_all



def train_semi_center_loss(data_loader, model, optimizer, loss_func, device):

    ce_loss_func = loss_func['weighted_ce_loss_func']
    center_loss_func = loss_func['center_loss_func']  
    mse_loss_sum_func = loss_func['mse_loss_sum_func']
    base_optim = optimizer['base_optim']
    center_optim = optimizer['center_optim']
    
    alpha = 10
    model.train()
    loss_all, ce_losses, center_losses, mse_losses = 0., 0., 0., 0.
    
    for i, (data, unsup_data) in enumerate(data_loader):
        
        data = data.to(device)        
        unsup_data = unsup_data.to(device)
        
        c = one_hot_encoding(data.y).to(device)
        l_enc, g_enc, g_out, l_dec = model(data, c)
        ce_loss = ce_loss_func(g_out, data.y)
        center_loss = center_loss_func(g_enc, data.y)
         
        unsup_l_enc, unsup_g_enc = model.encoder(unsup_data)
        unsup_l_dec = model.decoder(unsup_l_enc)

#         num_classes = center_loss_func.centers.size(0)
#         num_unsup = unsup_g_enc.size(0)
#         unsup_g_enc_ = unsup_g_enc.unsqueeze(0).repeat(num_classes, 1, 1)
#         centers_ = center_loss_func.centers.detach().unsqueeze(1).repeat(1, num_unsup, 1)
#         mse = torch.sum((unsup_g_enc_ - centers_) ** 2, dim=-1)
#         unsup_y = torch.argmin(mse, dim=0)
#         unsup_c = one_hot_encoding(unsup_y).to(device)

        x = data.x[:, :model.seq_len * model.feat_size]
        x = x.view(-1, model.num_nodes, model.seq_len, model.feat_size)     
        unsup_x = unsup_data.x[:, :model.seq_len * model.feat_size]
        unsup_x = unsup_x.view(-1, model.num_nodes, model.seq_len, model.feat_size)     
        
        mse_loss = mse_loss_sum_func(torch.cat((x, unsup_x), dim=0), 
                                     torch.cat((l_dec, unsup_l_dec), dim=0))   

        loss = mse_loss / 1000
#         loss += ce_loss
        loss += center_loss * 10

        base_optim.zero_grad()
        center_optim.zero_grad()           
        loss.backward()
        base_optim.step()
        
        # multiple (1./alpha) in order to remove the effect of alpha on updating centers
        for param in center_loss_func.parameters():
            param.grad.data *= (1. / alpha)
        center_optim.step()
        
        ce_losses += ce_loss.item()
        mse_losses += mse_loss.item()
        center_losses += center_loss.item()
        loss_all += loss.item()

    loss_dict = {
        'Total Loss': loss_all,
        'Reconstruct Loss': mse_losses,
        'Center Loss': center_losses,
    }
    verbose(loss_dict)
    return loss_all