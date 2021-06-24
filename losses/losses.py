import torch
import torch.nn as nn
import torch.nn.functional as F

def beta_vae_loss(reco_x, x, logvar, mu, beta, type='L2'):
    if type == 'BCE':
        reco_x_loss = F.binary_cross_entropy(reco_x, x, reduction='sum')
    elif type == 'L1':
        reco_x_loss = F.l1_loss(reco_x, x, size_average=False)
    else:
        reco_x_loss = F.mse_loss(reco_x, x, size_average=False)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return (reco_x_loss + beta*kld)/x.shape[0]

def KL_divergence(logvar, mu):
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
    return kld.mean()

def dice_loss(pred, target):
    """
    This definition generalize to real valued pred and target vector.
    This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """
    smooth = 0.1 #1e-12

    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    #A_sum = torch.sum(tflat * iflat)
    #B_sum = torch.sum(tflat * tflat)
    loss = ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)).mean()
    
    return 1 - loss

def HSIC_lossfunc(x, y):
    assert x.dim() == y.dim() == 2
    assert x.size(0) == y.size(0)
    m = x.size(0)
    h = torch.eye(m) - 1/m
    h = h.to(x.device)
    K_x = gaussian_kernel(x)
    K_y = gaussian_kernel(y)
    return K_x.mm(h).mm(K_y).mm(h).trace() / (m-1+1e-10)


def gaussian_kernel(x, y=None, sigma=5):
    if y is None:
        y = x
    assert x.dim() == y.dim() == 2
    assert x.size() == y.size()
    z = ((x.unsqueeze(0) - y.unsqueeze(1)) ** 2).sum(-1)
    return torch.exp(- 0.5 * z / (sigma * sigma))

def LS_dis(score_real, score_fake):
    return 0.5 * (torch.mean((score_real-1)**2) + torch.mean(score_fake**2))

def LS_model(score_fake):
    return 0.5 * (torch.mean((score_fake-1)**2))
