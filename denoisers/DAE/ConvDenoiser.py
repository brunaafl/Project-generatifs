"""
From https://github.com/Garima13a/Denoising-Autoencoder
"""

import torch.nn as nn

class ConvDenoiser(nn.Module):
    def __init__(self, in_channels, encoder_layers = [32, 16, 8], decoder_layers=[16, 32], sigmoid=False):
        super(ConvDenoiser, self).__init__()

        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers

        # Encoder
        n_input = in_channels
        encoder = []
        for n_layer in self.encoder_layers:
            encoder.append(nn.Conv2d(n_input, n_layer, 3, padding=1))
            encoder.append(nn.ReLU())
            encoder.append(nn.MaxPool2d(2, 2))
            n_input = n_layer

        self.encoder = nn.Sequential(*encoder)

        # Decoder
        n_input = in_channels
        decoder = []
        decoder.append(nn.ConvTranspose2d(8, 8, 3, stride=2))
        decoder.append(nn.ReLU())
        for n_layer in self.decoder_layers:
            encoder.append(nn.ConvTranspose2d(n_input, n_layer, 2, stride=2))
            encoder.append(nn.ReLU())
            n_input = n_layer
        decoder.append(nn.Conv2d(32, in_channels, 3, padding=1))
        decoder.append(nn.Sigmoid() if sigmoid else nn.Identity())
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
