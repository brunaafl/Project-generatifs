from matplotlib import pyplot as plt

from util.img_utils import clear_color


def plot_x_z_reconstruction(Z_step, X_step, step):
    # At the step {step} of a gibbs sampling pipeline
    # Plot estimated X and Z
    fig, axes = plt.subplots(1, 2, figsize=(20, 20))

    axes[0].imshow(clear_color(X_step))
    axes[0].set_title(f'X Reconstructed image - step {step}')
    axes[0].axis('off')

    axes[1].imshow(clear_color(Z_step))
    axes[1].set_title('Z Reconstructed image - step {step}')
    axes[1].axis('off')

    return fig, axes

