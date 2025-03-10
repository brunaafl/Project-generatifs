import torch

from denoisers.DAE.ConvDenoiser import ConvDenoiser
from denoisers.DnCNN.model import DnCNN


def create_model_ConvDenoiser(
    in_channels,
    encoder_layers,
    decoder_layers,
    sigmoid,
    model_path='',
):

    checkpoint_path = '/PnP-SGS-project/models/convdenoiser50.pt'
    model = ConvDenoiser(in_channels, encoder_layers, decoder_layers, sigmoid)
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'), strict=False)
    model.eval()

    return model

