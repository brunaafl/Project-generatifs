import torch

from denoisers.hierarquicalVAE.hVAE import NVAE


def create_model_nvae(
    z_dim=512,
    img_dim=(64, 64),
    model_path='',
):

    checkpoint_path = '/PnP-SGS-project/models/nvae_simple_loss_epoch_199.pth'
    model = NVAE(z_dim=z_dim, img_dim=img_dim)
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'), strict=False)
    model.eval()
    return model
