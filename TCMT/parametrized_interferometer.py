import torch
from torch.linalg import matrix_exp
from typing import List, Union

def hermitian_from_parameters(size: int, parameters: Union[List[float], torch.Tensor]) -> torch.Tensor:
    """Constructs an n x n Hermitian matrix with trace 0."""
    if len(parameters) != size**2 - 1:
        raise ValueError(f"Expected {size**2 - 1} parameters, got {len(parameters)}.")

    parameters = torch.tensor(parameters, dtype=torch.float32) if not isinstance(parameters, torch.Tensor) else parameters
    H = torch.zeros((size, size), dtype=torch.complex64)
    idx = 0

    for i in range(size):
        if i < size - 1:
            diag_mask = torch.zeros((size, size), dtype=torch.complex64)
            diag_mask[i, i] = 1
            H = H + diag_mask * parameters[idx]
            idx += 1
        for j in range(i + 1, size):
            real, imag = parameters[idx], parameters[idx + 1]
            real_mask = torch.zeros((size, size), dtype=torch.complex64)
            imag_mask = torch.zeros((size, size), dtype=torch.complex64)

            real_mask[i, j] = real_mask[j, i] = 1
            imag_mask[i, j], imag_mask[j, i] = 1j, -1j

            H = H + real_mask * real + imag_mask * imag
            idx += 2

    diag_mask = torch.zeros((size, size), dtype=torch.complex64)
    diag_mask[size - 1, size - 1] = 1
    H = H + diag_mask * -torch.sum(torch.diag(H[:-1]))
    return H

def unitary_from_parameters(size: int, parameters: Union[List[float], torch.Tensor]) -> torch.Tensor:
    """Constructs an n x n unitary matrix."""
    H = hermitian_from_parameters(size, parameters)
    return matrix_exp(1j * H)

def random_unitary(size: int) -> torch.Tensor:
    """Generates a random n x n unitary matrix."""
    return unitary_from_parameters(size, torch.rand(size**2 - 1))

def is_hermitian(matrix: torch.Tensor) -> bool:
    """Checks if a matrix is Hermitian."""
    return torch.allclose(matrix, matrix.conj().T)

def is_unitary(matrix: torch.Tensor) -> bool:
    """Checks if a matrix is unitary."""
    I = torch.eye(matrix.shape[0], dtype=matrix.dtype)
    return torch.allclose(matrix @ matrix.conj().T, I, atol=1e-6, rtol=1e-6)

if __name__ == "__main__":
    n = 3
    params = torch.rand(n**2 - 1)

    U = unitary_from_parameters(n, params)
    print("Unitary Matrix:", torch.real(U).round(decimals=3) + 1j * torch.imag(U).round(decimals=3), sep="\n")
    print("Is Unitary:", is_unitary(U))

    H = hermitian_from_parameters(n, params)
    print("\nHermitian Matrix:", torch.real(H).round(decimals=3) + 1j * torch.imag(H).round(decimals=3), sep="\n")
    print("Is Hermitian:", is_hermitian(H))

    print("\nEigenvalues of Hermitian Matrix:", torch.linalg.eigvals(H))

    rand_U = random_unitary(n)
    print("\nRandom Unitary Matrix:", torch.real(rand_U).round(decimals=3) + 1j * torch.imag(rand_U).round(decimals=3), sep="\n")
    print("Is Random Unitary:", is_unitary(rand_U))
