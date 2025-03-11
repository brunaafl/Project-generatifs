import torch
from matplotlib import pyplot as plt


def create_sampler_dn(tau, nu, s):
    return DnCNNsampler(tau=tau, nu=nu, s=s)

class DnCNNsampler:
    def __init__(self, tau, nu, s):
        self.tau = tau
        self.nu = nu
        self.s = s
        self.residuals = []

    def set_y(self, y):
        self.y = y
        self.normxinit = torch.linalg.vector_norm(y)

    def fidelity(self, x, operator):
        Hx = operator(x)
        return torch.sum((Hx - self.y) ** 2) / (2 * self.nu ** 2)

    def plot_residuals(self):
        plt.semilogy(self.residuals)
        plt.title('Residual Norm')
        plt.show()

    def sample(self, x, model, operator):
        x_t = x.clone().requires_grad_(True)
        fx = self.fidelity(x_t, operator)

        grad_fx = torch.autograd.grad(fx, x_t)[0]
        with torch.no_grad():
            x_aux = x_t - self.tau * grad_fx
            Dx = model(x_aux, sigma=self.s)
            x = Dx

        res = torch.linalg.vector_norm(x - x_t) / self.normxinit
        self.residuals.append(res.cpu())

        return x
