import math

import torch
from tqdm import tqdm
import numpy as np

from guided_diffusion.gaussian_diffusion import betas_for_alpha_bar
from util.plot_utils import plot_x_z_reconstruction


def get_named_beta_schedule(schedule_name="linear", num_diffusion_timesteps=1000):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")

class GibbsSampler:
    def __init__(self, Y, sigma, operator, model, model_type, device, sampler=None,
                 N_MC=23, N_bi=20, rho=0.1, rho_decay_rate=0.8, plot_process=None):

        if sampler is None and model_type == 'DDMP':
            ValueError('sampler is required for DDMP model')

        self.N_MC = N_MC
        self.N_bi = N_bi
        self.rho = rho
        self.rho_decay_rate = rho_decay_rate
        self.device = device
        self.Y = Y
        self.sigma = sigma
        self.operator = operator
        self.model = model
        self.model_type = model_type
        self.sampler = sampler
        self.plot_process = plot_process


        # Get alphas using the beta schedule
        betas = get_named_beta_schedule('linear', 1000)
        self.alphas = np.cumsum(betas) / np.max(np.cumsum(betas))

        # Initialize matrices to store iterates
        self.X_MC = torch.zeros(size=(3, 256, 256, N_MC+1), device=self.device)
        self.Z_MC = torch.zeros(size=(3, 256, 256, N_MC+1), device=self.device)

        # Initialize X_MC and Z_MC with random values
        self.X_MC[:,:,:,0] = torch.randn((3, 256, 256), device=self.device)
        self.Z_MC[:,:,:,0] = torch.randn((3, 256, 256), device=self.device)

    def estimate_time(self, value, array=None):
        if array is None:
            array = self.alphas
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    def compute_last_diff_step(self, t_start, t):
        if t < self.N_bi:
            t_stop = int(t_start * 0.7)
        else:
            t_stop = 0
        return t_stop

    def run(self):
        for t in tqdm(range(self.N_MC)):
            # Likelihood step
            self.X_MC[:,:,:,t+1] = self.operator.proximal_generator(
                self.Z_MC[:,:,:,t], self.Y, self.sigma, rho=self.rho
            ) # sigma is related to p(y|x)

            # Update rho and time step
            # Estimating noise level
            rho_iter = self.rho * (self.rho_decay_rate ** t)
            t_start = self.estimate_time(rho_iter)
            t_stop = self.compute_last_diff_step(t_start, t)

            # Prior step
            if self.model_type == 'DDMP':
                self.Z_MC[:,:,:,t+1] = self.sampler.diffuse_back(
                    x=self.X_MC[:,:,:,t+1].unsqueeze(0),
                    model=self.model,
                    t_start=1000 - t_start,
                    t_end=1000 - t_stop
                ).squeeze(0)
            elif self.model_type == 'DnCNN':
                self.Z_MC[:,:,:,t+1] = self.model(self.X_MC[:,:,:,t+1].unsqueeze(0))

            if self.plot_process is not None:
                if (t % self.plot_process == 0) and (t < self.N_bi):
                    # plot the reconstruction on the step t
                    plot_x_z_reconstruction(self.Z_MC[:,:,:,t+1], self.X_MC[:,:,:,t+1], t)


        return self.X_MC, self.Z_MC
