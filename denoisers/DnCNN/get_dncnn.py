import torch

from denoisers.DnCNN.model import DnCNN


def create_model_DnCNN(
    channels=3,
    num_of_layers=17,
    model_path='',
):

    checkpoint_path = '/PnP-SGS-project/models/DnCNN_net.pth'
    model = DnCNN(channels=channels, num_of_layers=num_of_layers)
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'), strict=False)
    model.eval()

    return model

