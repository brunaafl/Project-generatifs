import torch
from matplotlib import pyplot as plt
from scipy.fft import fft2

from util.img_utils import psnr, clear_color


def create_sampler_dn(tau, nu, s, operator, s_decay=0.8, normalize=True):
    return DnCNNsampler(tau=tau, nu=nu, s=s, operator=operator, s_decay=s_decay, normalize=normalize)

class DnCNNsampler:
    def __init__(self, tau, nu, s, operator, s_decay=0.8, normalize=True):
        self.tau = tau
        self.nu = nu
        self.s = s
        self.operator = operator
        self.s_decay = s_decay
        self.normalize = normalize

        self.residuals = []
        self.psnr = []

    def set_y_x0(self, y, x0):
        self.y = y.clone()
        self.normxinit = torch.linalg.vector_norm(y)

        self.x0 = x0

    def fidelity(self, x):
        Hx = self.operator.forward(x)
        return torch.sum((Hx - self.y) ** 2) / (2 * self.nu ** 2)

    def plot_residuals(self):
        fig, axs = plt.subplots()
        axs.semilogy(self.residuals)
        plt.title('Residual Norm')
        plt.show()

    def sample(self, x, model, niter=100, plot=False):
        s = self.s

        for it in range(niter):

            s = s * self.s_decay ** it

            x_t = x.clone().requires_grad_(True)
            fx = self.fidelity(x_t)

            grad_fx = torch.autograd.grad(fx, x_t)[0]
            with torch.no_grad():
                x_aux = x_t - self.tau * grad_fx
                Dx = model(x_aux, sigma=s)
                x = Dx

            res = torch.linalg.vector_norm(x - x_t) / self.normxinit
            self.residuals.append(res.detach().item())
            self.psnr.append(psnr(self.x0,x))

            if plot:
                if it % 10 == 0:
                    fig, ax = plt.subplots()
    
                    ax.imshow(clear_color(x,normalize=self.normalize))
                    ax.axis('off')
                    plt.show()

        return x
