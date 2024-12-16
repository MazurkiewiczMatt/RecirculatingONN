import torch
from torch.linalg import inv
from torch import nn
from torchdiffeq import odeint
import numpy as np

from .parametrized_interferometer import unitary_from_parameters

def time_evolution_operator(A, omega, kappa, U, nonlinearity):
    """Builds the time evolution operator."""
    I = torch.eye(U.shape[0], dtype=torch.complex64)
    mix = U.T @ U @ inv(I - U.T @ U + 1e-8 * I)
    freq = 1j * (omega[None, :] + nonlinearity[None, :] * torch.abs(A)**2)
    T1 = torch.diag_embed(freq)
    T2 = -kappa * (0.5 * torch.eye(U.shape[0]) + mix)
    T2 = T2.unsqueeze(0)
    A = A.unsqueeze(-1)
    dydt = torch.matmul(T1 + T2, A).squeeze(-1)
    return dydt

def create_ode(omega, kappa, U, nonlinearity):
    def ode(t, A):
        return time_evolution_operator(A, omega, kappa, U, nonlinearity)
    return ode

class Circuit(nn.Module):
    """Photonic circuit model."""
    def __init__(self, modes, input_modes, omega, kappa, nonlinearity, params=None):
        super().__init__()
        self.modes = modes
        self.input_modes = input_modes
        self.omega, self.kappa, self.nonlinearity = omega, kappa, nonlinearity
        self.params = nn.Parameter(torch.rand(modes**2 - 1) if params is None else params)

    def forward(self, A0):
        padded_input = torch.zeros(A0.size(0), self.modes, dtype=A0.dtype, device=A0.device)
        padded_input[:, :self.input_modes] = A0
        if self.input_modes < self.modes:
            padded_input[:, self.input_modes:] = torch.ones(A0.size(0), self.modes-self.input_modes, dtype=A0.dtype, device=A0.device)
        U = unitary_from_parameters(self.modes, self.params)
        ode_func = create_ode(self.omega, self.kappa, U, self.nonlinearity)
        t_span = (0, 2)
        eval_pts = 200
        t_eval = torch.linspace(*t_span, eval_pts)
        return odeint(ode_func, padded_input, t_eval, method='dopri5', rtol=1e-7, atol=1e-9)

    @classmethod
    def new(cls, modes, input_modes):
        """Generates a circuit with random parameters."""
        omega = torch.full((modes,), 5.0)
        kappa = torch.full((modes,), 5.0)
        nonlinearity = torch.full((modes,), 0.2)
        params = (torch.rand(modes**2 - 1)-0.5)*2*np.pi

        return cls(modes, input_modes, omega, kappa, nonlinearity, params)
