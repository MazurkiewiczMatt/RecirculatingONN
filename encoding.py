import pandas as pd
import numpy as np
import torch


class Encoding:
    """
    Base class for data encodings.

    Methods:
        encode(data):
            Encodes the input data.
        loss_function():
            Returns the loss function for the encoding.
    """

    def encode(self, data):
        raise NotImplementedError("Subclasses must implement the encode method.")

    def loss_function(self):
        raise NotImplementedError("Subclasses must implement the loss_function method.")


class PhaseEncoding(Encoding):
    """
    Phase encoding class for converting data to a cavity-compatible format.
    """

    def encode(self, data):
        if isinstance(data, pd.Series):
            normalized_data = self._normalize_series(data)
        elif isinstance(data, pd.DataFrame):
            normalized_data = data.apply(self._normalize_series)
        else:
            raise ValueError("Input must be a pandas Series or DataFrame.")

        normalized_data = (
            normalized_data.to_numpy() if isinstance(data, pd.DataFrame)
            else normalized_data.values
        )
        encoded = np.exp(1j * normalized_data)
        return torch.tensor(encoded, dtype=torch.cfloat)

    def _normalize_series(self, series):
        if series.dtype.kind in 'biufc':  # Numeric data
            return (series - series.min()) / (series.max() - series.min()) * (2 * np.pi)
        else:  # Categorical data
            unique_values = sorted(series.unique())
            mapping = {val: i * (2 * np.pi) / len(unique_values) for i, val in enumerate(unique_values)}
            return series.map(mapping)

    def loss_function(self):
        def complex_mse_loss(output, target):
            """Computes the mean squared error for complex-valued data."""
            return (torch.abs(output - target) ** 2).mean()
        return complex_mse_loss


class CategoricalEncoding(Encoding):
    """
    Categorical encoding for one-hot encoding of categorical data in Series or DataFrames.
    """

    def encode(self, data):
        if isinstance(data, pd.Series):
            data = data.to_frame()

        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input must be a pandas Series or DataFrame.")

        # Ensure all columns are categorical
        if not all(data[col].dtype.kind not in 'biufc' for col in data.columns):
            raise ValueError("All columns in the input must be categorical.")

        one_hot_encoded = pd.get_dummies(data, dtype=np.float32)
        return torch.tensor(one_hot_encoded.values, dtype=torch.float32)

    def loss_function(self):
        def loss(output, target):
            norm_squared = torch.abs(output) ** 2
            softmaxed = torch.nn.functional.softmax(norm_squared, dim=1)
            return torch.nn.functional.cross_entropy(softmaxed, target.argmax(dim=1))

        return loss





def cavity_encoding(data, encoding_scheme):
    """
    Encodes data into a cavity-compatible format using the specified encoding scheme.

    Parameters:
        data: pd.Series or pd.DataFrame
            Input data to encode.
        encoding_scheme: str
            Encoding scheme to use (e.g., 'phase').

    Returns:
        encoded: torch.Tensor
            Encoded data as a complex tensor.
        loss_function: callable
            Loss function appropriate for the encoding scheme.
    """
    encodings = {
        'phase': PhaseEncoding(),
        'categorical': CategoricalEncoding()
    }

    if encoding_scheme not in encodings:
        raise NotImplementedError(f"Encoding scheme '{encoding_scheme}' is not supported.")

    encoder = encodings[encoding_scheme]
    return encoder.encode(data), encoder.loss_function()
