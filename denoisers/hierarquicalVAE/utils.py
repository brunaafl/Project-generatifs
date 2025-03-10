# utils functions for training and sampling:
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


def kl(mu, log_var):
    """
    kl loss with standard norm distribute
    :param mu:
    :param log_var:
    :return:
    """
    loss = -0.5 * torch.sum(1 + log_var - mu ** 2 - torch.exp(log_var), dim=[1, 2, 3])
    return torch.mean(loss, dim=0)


def kl_delta(delta_mu, delta_log_var, mu, log_var):
    var = torch.exp(log_var)
    delta_var = torch.exp(delta_log_var)

    loss = -0.5 * torch.sum(1 + delta_log_var - delta_mu ** 2 / var - delta_var, dim=[1, 2, 3])
    return torch.mean(loss, dim=0)


def reparameterize(mu, std):
    z = torch.randn_like(mu) * std + mu
    return z


def add_sn(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        return spectral_norm(m)
    else:
        return m

