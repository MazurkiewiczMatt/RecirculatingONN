import pandas as pd
import numpy as np
import torch

class Encoding:
    """
    Base class for data encodings.

    Methods:
        encode(data):
            Encodes the input data into a transformed representation.
    """

    def encode(self, data):
        raise NotImplementedError("Subclasses must implement the encode method.")


class Decoding:
    """
    Base class for data decodings.

    Methods:
        decode(encoded_data):
            Decodes the transformed data back into the original data space.
        loss_function():
            Returns the loss function for evaluating the reconstruction or classification.
    """

    def decode(self, encoded_data):
        raise NotImplementedError("Subclasses must implement the decode method.")

    def loss_function(self):
        raise NotImplementedError("Subclasses must implement the loss_function method.")


class PhaseEncoding(Encoding, Decoding):
    """
    Phase encoding class for converting data to a cavity-compatible format.
    Encodes numeric or categorical data into complex phase values on the unit circle.
    """

    def __init__(self):
        super().__init__()
        self.data_min = None
        self.data_max = None
        self.unique_values = None
        self.is_categorical = False

    def encode(self, data):
        self.is_categorical = False

        if isinstance(data, pd.Series):
            normalized_data = self._normalize_series(data)
        elif isinstance(data, pd.DataFrame):
            # Store per-column metadata for decoding
            # We'll store a dict of (min, max, unique_values) per column
            self.column_metadata = {}
            normalized_data = pd.DataFrame(index=data.index)
            for col in data.columns:
                col_series = data[col]
                if col_series.dtype.kind in 'biufc':
                    col_min = col_series.min()
                    col_max = col_series.max()
                    self.column_metadata[col] = {
                        'type': 'numeric',
                        'min': col_min,
                        'max': col_max
                    }
                    normalized_col = (col_series - col_min) / (col_max - col_min) * (2 * np.pi)
                    normalized_data[col] = normalized_col
                else:
                    unique_vals = sorted(col_series.unique())
                    mapping = {val: i*(2*np.pi)/len(unique_vals) for i, val in enumerate(unique_vals)}
                    self.column_metadata[col] = {
                        'type': 'categorical',
                        'unique_values': unique_vals,
                        'mapping': mapping
                    }
                    normalized_data[col] = col_series.map(mapping)
        else:
            raise ValueError("Input must be a pandas Series or DataFrame.")

        if isinstance(data, pd.Series):
            if data.dtype.kind in 'biufc':  # Numeric data
                self.data_min = data.min()
                self.data_max = data.max()
            else:  # Categorical
                self.is_categorical = True
                self.unique_values = sorted(data.unique())
        encoded = np.exp(1j * normalized_data.to_numpy()) if isinstance(normalized_data, pd.DataFrame) \
                  else np.exp(1j * normalized_data.values)
        return torch.tensor(encoded, dtype=torch.cfloat)

    def _normalize_series(self, series):
        if series.dtype.kind in 'biufc':  # Numeric data
            col_min = series.min()
            col_max = series.max()
            self.data_min = col_min
            self.data_max = col_max
            return (series - col_min) / (col_max - col_min) * (2 * np.pi)
        else:  # Categorical data
            unique_values = sorted(series.unique())
            self.unique_values = unique_values
            mapping = {val: i * (2 * np.pi) / len(unique_values) for i, val in enumerate(unique_values)}
            self.is_categorical = True
            return series.map(mapping)

    def decode(self, encoded_data):
        # encoded_data is a complex tensor on the unit circle
        # We must retrieve original data space from the phase angle.
        # phase angles are in [-pi, pi], but we normalized data into [0, 2*pi].
        angles = torch.angle(encoded_data).numpy()  # shape (n, [m])

        # For Series (single dimension)
        if hasattr(self, 'unique_values') and self.unique_values is not None and not hasattr(self, 'column_metadata'):
            # Single-column scenario
            if self.is_categorical:
                # Map angles back to categories
                # Since data was mapped from [0, 2*pi], angle might be negative, convert to [0,2*pi)
                angles = (angles + 2*np.pi) % (2*np.pi)
                interval = 2*np.pi/len(self.unique_values)
                idx = np.floor(angles/interval).astype(int)
                idx = np.clip(idx, 0, len(self.unique_values)-1)
                return pd.Series([self.unique_values[i] for i in idx])
            else:
                # Numeric
                # angles back to [0, 2*pi], then scale back to original range
                angles = (angles + 2*np.pi) % (2*np.pi)
                original = (angles / (2*np.pi)) * (self.data_max - self.data_min) + self.data_min
                return pd.Series(original)

        # For DataFrame (multi-column scenario)
        if hasattr(self, 'column_metadata'):
            df_decoded = pd.DataFrame(index=range(angles.shape[0]))
            col_idx = 0
            for col, meta in self.column_metadata.items():
                col_angles = angles[:, col_idx]
                col_angles = (col_angles + 2*np.pi) % (2*np.pi)
                if meta['type'] == 'numeric':
                    col_min = meta['min']
                    col_max = meta['max']
                    col_original = (col_angles / (2*np.pi)) * (col_max - col_min) + col_min
                else:
                    # Categorical
                    unique_values = meta['unique_values']
                    interval = 2*np.pi/len(unique_values)
                    idx = np.floor(col_angles/interval).astype(int)
                    idx = np.clip(idx, 0, len(unique_values)-1)
                    col_original = [unique_values[i] for i in idx]
                df_decoded[col] = col_original
                col_idx += 1
            return df_decoded

        raise ValueError("Insufficient metadata to decode the data.")

    def loss_function(self):
        def complex_mse_loss(output, target):
            """Computes the mean squared error for complex-valued data."""
            return (torch.abs(output - target) ** 2).mean()
        return complex_mse_loss


class CategoricalEncoding(Encoding, Decoding):
    """
    Categorical encoding for one-hot encoding of categorical data in Series or DataFrames.
    """

    def __init__(self):
        super().__init__()
        self.categories = None

    def encode(self, data):
        if isinstance(data, pd.Series):
            data = data.to_frame()

        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input must be a pandas Series or DataFrame.")

        # Ensure all columns are categorical
        for col in data.columns:
            if data[col].dtype.kind in 'biufc':
                raise ValueError("All columns in the input must be categorical.")

        # Store categories for decoding
        self.categories = {}
        for col in data.columns:
            self.categories[col] = sorted(data[col].unique())

        one_hot_encoded = pd.get_dummies(data, dtype=np.float32)
        return torch.tensor(one_hot_encoded.values, dtype=torch.float32)

    def decode(self, encoded_data):
        # Decode one-hot encoded data back to categories
        # We know that the original DataFrame columns were all categorical.
        # We must reassemble them. We'll have to guess the original column structure.
        # Since we stored categories per column, we can reconstruct columns by distributing
        # the encoded data into groups corresponding to each column's categories.

        encoded_np = encoded_data.numpy()
        df_decoded = pd.DataFrame()

        start_idx = 0
        for col, cats in self.categories.items():
            end_idx = start_idx + len(cats)
            col_data = encoded_np[:, start_idx:end_idx]
            # One-hot: index of max value
            cat_indices = np.argmax(col_data, axis=1)
            df_decoded[col] = [cats[i] for i in cat_indices]
            start_idx = end_idx

        return df_decoded

    def loss_function(self):
        def loss(output, target):
            # target must be in same encoded form, so we find the max along dim=1 to get indices.
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
            Encoding scheme to use (e.g., 'phase', 'categorical').

    Returns:
        encoded: torch.Tensor
            Encoded data.
        decoder: Decoding instance
            An instance of the class used for decoding (with a loss function).
    """
    schemes = {
        'phase': PhaseEncoding(),
        'categorical': CategoricalEncoding()
    }

    if encoding_scheme not in schemes:
        raise NotImplementedError(f"Encoding scheme '{encoding_scheme}' is not supported.")

    codec = schemes[encoding_scheme]
    encoded = codec.encode(data)
    # codec now holds the metadata needed to decode and the loss function
    return encoded, codec
