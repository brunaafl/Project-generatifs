import torch
from matplotlib import pyplot as plt
import seaborn as sns

from util.img_utils import clear_color


def show_reconstruction(Z_step, X_step, step, normalize=True):
    # At the step {step} of a gibbs sampling pipeline
    # Plot estimated X and Z
    fig, axes = plt.subplots(1, 2, figsize=(20, 20))

    axes[0].imshow(clear_color(X_step, normalize))
    axes[0].set_title(f'X Reconstructed image - step {step}')
    axes[0].axis('off')

    axes[1].imshow(clear_color(Z_step, normalize))
    axes[1].set_title(f'Z Reconstructed image - step {step}')
    axes[1].axis('off')

    return fig, axes

def confidence_interval(data, C=0.9):

    # Percentiles of the gaussian
    print(data.shape)
    X_lower = torch.quantile(data,(1-C)/2, dim=-1).T.detach().cpu().numpy()
    print(X_lower.shape)
    X_upper = torch.quantile(data, (C+1)/2, dim=-1).T.detach().cpu().numpy()
    X_std = torch.std(data, dim=-1).T.detach().cpu().numpy()

    # Heatmap of 95% CI
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    CI = 100 * C
    im1 = ax[0].imshow(X_upper - X_lower, cmap='magma')
    ax[0].set_title(f"{CI}% Confidence Interval")
    ax[0].axis("off")
    fig.colorbar(im1, ax=ax[0])

    im2 = ax[1].imshow(X_std, cmap='inferno')
    ax[1].set_title("Standard Deviation on each pixel")
    ax[1].axis("off")
    fig.colorbar(im2, ax=ax[1])

    plt.tight_layout()
    plt.show()
