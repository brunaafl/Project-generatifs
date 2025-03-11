import torch
from matplotlib import pyplot as plt

from util.img_utils import clear_color


def show_reconstruction(Z_step, X_step, step):
    # At the step {step} of a gibbs sampling pipeline
    # Plot estimated X and Z
    fig, axes = plt.subplots(1, 2, figsize=(20, 20))

    axes[0].imshow(clear_color(X_step))
    axes[0].set_title(f'X Reconstructed image - step {step}')
    axes[0].axis('off')

    axes[1].imshow(clear_color(Z_step))
    axes[1].set_title(f'Z Reconstructed image - step {step}')
    axes[1].axis('off')

    return fig, axes

def confidence_interval(data, C=0.9):

    # Percentiles of the gaussian
    X_lower = torch.quantile(data,(1-C)/2, dim=0)
    X_upper = torch.quantile(data, (C+1)/2, dim=0)
    X_std = torch.std(data, dim=0)

    fig, ax = plt.subplots(1,2)

    # Heatmap of 95% CI
    ax[0].imshow(X_upper - X_lower, cmap='magma')
    ax[0].title(f"{int(100*C)}% Confidence Interval")
    ax[0].axis("off")

    # standard deviation
    ax[1].imshow(X_std, cmap='inferno')
    ax[1].title("Standard Deviation on each pixel")
    ax[1].axis("off")

    plt.tight_layout()
    plt.show()
