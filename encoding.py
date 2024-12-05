import pandas as pd
import numpy as np
import torch

def cavity_encoding(x, encoding_scheme):
    x_normalized = normalize(x)

    if isinstance(x, pd.DataFrame):
        x_normalized = x_normalized.to_numpy()
    elif isinstance(x, pd.Series):
        x_normalized = x_normalized.values

    # Apply encoding
    if encoding_scheme == 'phase':
        encoded = np.exp(1j * x_normalized)
    else:
        raise NotImplementedError(f'{encoding_scheme} is not supported.')

    # Convert to PyTorch tensor
    encoded_tensor = torch.tensor(encoded, dtype=torch.cfloat)
    return encoded_tensor

def normalize(x):
    """
    Normalizes numerical columns from 0 to 2*pi.
    Maps discrete columns onto equidistant points in the same range.
    """
    if isinstance(x, pd.Series):
        # Single column handling
        if x.dtype.kind in 'biufc':  # Numeric
            normalized = (x - x.min()) / (x.max() - x.min()) * (2 * np.pi)
            return normalized
        else:  # Categorical
            unique_values = sorted(x.unique())  # Sort to ensure consistent order
            num_classes = len(unique_values)
            mapping = {val: i * (2 * np.pi) / num_classes for i, val in enumerate(unique_values)}
            normalized = x.map(mapping)
            return normalized

    elif isinstance(x, pd.DataFrame):
        # DataFrame handling
        normalized_df = x.copy()
        for col in normalized_df.columns:
            if normalized_df[col].dtype.kind in 'biufc':  # Numeric
                normalized_df[col] = (normalized_df[col] - normalized_df[col].min()) / (
                            normalized_df[col].max() - normalized_df[col].min()) * (2 * np.pi)
            else:  # Categorical
                unique_values = sorted(normalized_df[col].unique())
                num_classes = len(unique_values)
                mapping = {val: i * (2 * np.pi) / num_classes for i, val in enumerate(unique_values)}
                normalized_df[col] = normalized_df[col].map(mapping)
        return normalized_df
    else:
        raise ValueError("Input must be a pandas Series or DataFrame.")