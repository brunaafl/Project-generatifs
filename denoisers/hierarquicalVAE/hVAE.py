import torch

import torch.nn as nn

from denoisers.hierarquicalVAE.decoder import Decoder
from denoisers.hierarquicalVAE.encoder import Encoder
from denoisers.hierarquicalVAE.utils import reparameterize, kl


class NVAE(nn.Module):

    def __init__(self, z_dim, img_dim):
        super().__init__()

        self.encoder = Encoder(z_dim)
        self.decoder = Decoder(z_dim)

    def forward(self, x):
        """

        :param x: Tensor. shape = (B, C, H, W)
        :return:
        """

        mu, log_var, xs = self.encoder(x)

        # (B, D_Z)
        z = reparameterize(mu, torch.exp(0.5 * log_var)) # sampling top latent variable

        decoder_output, kl_losses = self.decoder(z, xs)

        kl_losses = [kl(mu, log_var)]+kl_losses

        recon_loss = nn.MSELoss(reduction='sum')(decoder_output, x)/decoder_output.shape[0]

        return decoder_output, recon_loss, kl_losses

