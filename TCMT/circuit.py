import torch
from torch.linalg import inv
from torch import nn
from torchdiffeq import odeint
import numpy as np
from torchinfo import summary

from .parametrized_interferometer import unitary_from_parameters


def kappa_function(t, kappa_params):
    """
    Computes kappa, a vector of differentiable step functions at time t.

    Parameters:
    t (torch.Tensor): Time tensor.
    kappa_params (torch.Tensor): Flattened tensor of parameters (length 3N).

    Returns:
    torch.Tensor: Vector of step function values, kappa.
    """
    # Reshape kappa_params into (N, 3)
    kappa_params = kappa_params.view(-1, 3)

    # Extract parameters
    kappa_0 = kappa_params[:, 0] ** 2
    kappa_1 = kappa_params[:, 1] ** 2
    transition_time = kappa_params[:, 2] ** 2

    # Fixed sharpness
    sharpness = 100.0

    # Compute kappa
    kappa = kappa_0 + (kappa_1 - kappa_0) * torch.sigmoid(sharpness * (t - transition_time))

    return kappa


def time_evolution_operator(A, omega, kappa, U, nonlinearity, lambd):
    """Builds the time evolution operator."""
    I = torch.eye(U.shape[0], dtype=torch.complex64)
    mix = U.T @ U @ inv(I*(1+lambd) - U.T @ U)
    freq = 1j * (omega[None, :] + nonlinearity * torch.abs(A)**2)
    T1 = torch.diag_embed(freq)
    sqrt_kappa = torch.diag_embed((kappa**0.5).to(dtype=torch.complex64))
    T2 = - sqrt_kappa @ (0.5 * I + mix) @ sqrt_kappa
    T2 = T2.unsqueeze(0)
    A = A.unsqueeze(-1)
    operator = torch.matmul(T1 + T2, A).squeeze(-1)
    return operator

def create_ode(omega, kappa, U, nonlinearity, lambd, kappa_params=None):
    if kappa is None and kappa_params is None:
        raise ValueError('kappa and kappa_params cannot be both None (must be either fixed or DoF).')
    def ode(t, A):
        if kappa is not None:
            dydt = time_evolution_operator(A, omega, kappa**2, U, nonlinearity**2, lambd)
        else:
            dydt = time_evolution_operator(A, omega, kappa_function(t, kappa_params), U, nonlinearity ** 2, lambd)
        return dydt
    return ode

class Circuit(nn.Module):
    """Photonic circuit model."""
    def __init__(self, computation_time, modes, input_modes, omega, kappa, nonlinearity, lambd, params, learnable_omega, learnable_nonlinearity, learnable_coupling, learnable_interferometer, temporal_DOF):
        super().__init__()
        self.computation_time = computation_time
        self.modes = modes
        self.input_modes = input_modes
        self.temporal_DOF = temporal_DOF
        if learnable_omega:
            self.omega = nn.Parameter(omega)
        else:
            self.omega = omega
        if learnable_nonlinearity:
            self.nonlinearity = nn.Parameter(nonlinearity)
        else:
            self.nonlinearity = nonlinearity
        if learnable_coupling:
            self.kappa = nn.Parameter(kappa)
        else:
            self.kappa = kappa

        self.lambd = lambd

        if learnable_interferometer:
            self.params = nn.Parameter(torch.rand(modes**2 - 1) if params is None else params)
        else:
            self.params = params

    def forward(self, A0):
        padded_input = torch.zeros(A0.size(0), self.modes, dtype=A0.dtype, device=A0.device)
        padded_input[:, :self.input_modes] = A0
        if self.input_modes < self.modes:
            padded_input[:, self.input_modes:] = torch.ones(A0.size(0), self.modes-self.input_modes, dtype=A0.dtype, device=A0.device)
        U = unitary_from_parameters(self.modes, self.params)
        if self.temporal_DOF:
            kappa = None
            kappa_params = self.kappa
        else:
            kappa = self.kappa
            kappa_params = None
        ode_func = create_ode(self.omega, kappa, U, self.nonlinearity, self.lambd, kappa_params=kappa_params)
        t_span = (0, self.computation_time)
        eval_pts = 200
        t_eval = torch.linspace(*t_span, eval_pts)
        return odeint(ode_func, padded_input, t_eval, method='dopri5', rtol=1e-8, atol=1e-10)

    def enable_cavity_training(self):
        print("Enabling training of cavity parameters (pretraining complete)")
        self.omega = nn.Parameter(self.omega)
        self.nonlinearity = nn.Parameter(self.nonlinearity)
        self.kappa = nn.Parameter(self.kappa)
        summary(self)

    @classmethod
    def new(cls, params):
        """Generates a circuit with random parameters."""

        bias_cavities = params.get("bias_cavities", 0)
        num_input_modes = params.get("num_input_modes")
        num_output_modes = params.get("num_output_modes")
        learnable_omega = params.get("learnable_omega")
        learnable_nonlinearity = params.get("learnable_nonlinearity")
        learnable_coupling = params.get("learnable_coupling")
        learnable_interferometer = params.get("learnable_interferometer")
        lambd = params.get("regularization")
        computation_time = params.get("computation_time")

        initialization_omega = params.get("initialization_omega")
        initialization_nonlinearity = params.get("initialization_nonlinearity")
        initialization_coupling = params.get("initialization_coupling")



        if "cavityless_pretraining" in params and params["cavityless_pretraining"] is not None:
            learnable_omega = False
            learnable_nonlinearity = False
            learnable_coupling = False

        temporal_DOF = False
        if "temporal_DOF" in params:
            temporal_DOF = params["temporal_DOF"]

        num_modes = max(num_input_modes + bias_cavities, num_output_modes)

        if initialization_omega is not None:
            omega = torch.full((num_modes,), initialization_omega, dtype=torch.complex64)
        else:
            omega = torch.randn(num_modes)

        if initialization_nonlinearity is not None:
            nonlinearity = torch.tensor(initialization_nonlinearity)
        else:
            nonlinearity = torch.tensor(0.0*1j)

        if temporal_DOF:
            kappa = torch.rand(num_modes*3)
        else:
            if initialization_coupling is not None:
                kappa = torch.full((num_modes,), initialization_coupling, dtype=torch.complex64)
            else:
                kappa =  torch.rand(num_modes)

        params = (torch.rand(num_modes**2 - 1)-0.5)*2*np.pi

        circuit = cls(computation_time, num_modes, num_input_modes, omega, kappa, nonlinearity, lambd, params, learnable_omega, learnable_nonlinearity, learnable_coupling, learnable_interferometer, temporal_DOF)
        print("\nModel summary")
        summary(circuit)

        return circuit
