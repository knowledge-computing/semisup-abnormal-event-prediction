import logging
import torch
import torch.nn as nn
import torch.nn.functional as F


def global_global_cdist_batch_loss(g_enc, data):
    
    if torch.sum(data.y == 0) == 0 or torch.sum(data.y == 1) == 0:
        return 0.

    pos_dis, pos_count = 0., 0
    pos_dis_, pos_count_ = 0., 0
    neg_dis, neg_count = 0., 0

    MSE_loss = nn.MSELoss(reduction='sum')

    for i, yi in enumerate(data.y):
        for j, yj in enumerate(data.y):

            if i == j:
                continue

            if yi == 1 and yj == 1:
                pos_dis += MSE_loss(g_enc[i], g_enc[j])
#                 pos_dis += torch.cdist(g_enc[i].view(1, -1), g_enc[j].view(1, -1)).sum()
                pos_count += 1

            if yi == 0 and yj == 0:
                pos_dis_ += MSE_loss(g_enc[i], g_enc[j])
#                 pos_dis_ += torch.cdist(g_enc[i].view(1, -1), g_enc[j].view(1, -1)).sum()
                pos_count_ += 1

            if yi != yj:
                neg_dis += MSE_loss(g_enc[i], g_enc[j])
                # neg_dis += torch.cdist(g_enc[i].view(1, -1), g_enc[j].view(1, -1)).sum()
                neg_count += 1

    loss = 0.
    if pos_count > 0:
        loss += pos_dis / pos_count
    if pos_count_ > 0:
        loss += pos_dis_ / pos_count_
    if neg_count > 0:
        loss -= neg_dis / neg_count
    return loss


def cvae_loss(x, de_x, mu, log_var):
    """
    reconstruction + KL divergence losses summed over all elements in the batch
    see Appendix B from VAE paper: Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    """

    BCE = F.binary_cross_entropy_with_logits(de_x, x, reduction='sum')  # the paper use binary cross entropy as the reconstruction loss
    MSE_loss = nn.MSELoss(reduction='sum')
    MSE = MSE_loss(de_x, x)
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())  # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    return MSE, KLD, MSE + KLD


class CenterLoss(nn.Module):

    def __init__(self, num_classes, feat_dim, device):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim)).to(device)

    def forward(self, x, y):
        
        batch_size = x.size(0)
        dist_mat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                   torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        dist_mat = torch.addmm(dist_mat, x, self.centers.t(), beta=1, alpha=-2)

        classes = torch.arange(self.num_classes).long().to(self.device)
        y = y.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = y.eq(classes.expand(batch_size, self.num_classes))

        dist = dist_mat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        return loss


class ContrastiveCenterLoss(nn.Module):

    def __init__(self, num_classes, feat_dim, device, lambda_c=1.0):
        super(ContrastiveCenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim).to(device))
        self.lambda_c = lambda_c

    # may not work due to flowing gradient. change center calculation to exp moving avg may work.
    def forward(self, x, y):

        batch_size = x.size()[0]
        expanded_centers = self.centers.expand(batch_size, -1, -1)
        expanded_hidden = x.expand(self.num_classes, -1, -1).transpose(1, 0)
        dist_centers = (expanded_hidden - expanded_centers).pow(2).sum(dim=-1)
        dist_same = dist_centers.gather(1, y.unsqueeze(1))
        intra_dist = dist_same.sum()
        inter_dist = dist_centers.sum().sub(intra_dist)
        epsilon = 1e-6
        loss = (self.lambda_c / 2.0 / batch_size) * intra_dist / (inter_dist + epsilon) / 0.1
        return loss



class FocalLoss(nn.Module):

    def __init__(self, gamma, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.nll_loss = torch.nn.NLLLoss(weight=weight)

    def forward(self, logpt, target):
        """ input: [N, C], target: [N, ] """

        pt = torch.exp(logpt)
        logpt = (1 - pt) ** self.gamma * logpt
        loss = self.nll_loss(logpt, target)
        return loss
