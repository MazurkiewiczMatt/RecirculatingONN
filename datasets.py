import pandas as pd
import torch
from torch.utils.data import random_split
import numpy as np

class Data:
    def __init__(self, features, targets, original):
        self.features = features
        self.targets = targets
        self.original = original

class SpiralDataset:
    def __init__(self, data, metadata):
        self.data = data
        self.metadata = metadata

def fetch_spiral_dataset(n_samples=500, noise=0.2, angle_offset=np.pi):
    """
    Generates a 2D spiral dataset with two interleaved spirals.

    Parameters:
        n_samples (int): Number of samples per spiral.
        noise (float): Standard deviation of Gaussian noise added to the data.
        angle_offset (float): Angular offset (in radians) between the two spirals.

    Returns:
        SpiralDataset: An object containing the dataset and metadata.
    """
    def generate_spiral(n_points, noise, label, angle_offset=0):
        theta = np.sqrt(np.random.rand(n_points)) * 2 * np.pi  # Angle
        r = theta + noise * np.random.randn(n_points) + 0.3          # Radius with noise
        x = r * np.cos(theta + angle_offset)
        y = r * np.sin(theta + angle_offset)

        labels = np.full(n_points, str(label), dtype=str)
        return np.stack([x, y], axis=1), labels

    # Generate data for two spirals
    spiral_0, labels_0 = generate_spiral(n_samples, noise, label=0)
    spiral_1, labels_1 = generate_spiral(n_samples, noise, label=1, angle_offset=angle_offset)

    # Combine data
    X = np.vstack([spiral_0, spiral_1])
    y = np.hstack([labels_0, labels_1])

    # Create DataFrames
    feature_df = pd.DataFrame(X, columns=["x1", "x2"])
    target_df = pd.DataFrame(y, columns=["target"])

    # Combine features and targets into a Data object
    data = Data(features=feature_df, targets=target_df, original=pd.concat([feature_df, target_df], axis=1))

    # Metadata
    metadata = {
        "name": "spiral",
        "description": "2D synthetic dataset with two interleaved spirals",
        "n_samples": 2 * n_samples,
        "n_features": 2,
        "n_classes": 2,
        "abstract": "A synthetic dataset with two interleaved spirals",
    }

    return SpiralDataset(data=data, metadata=metadata)


def get_dataset(dataset_name):
    # available: "iris" (150 datapoints, 4 features); "tic-tac-toe" (958, 9); "car evaluation" (1728, 6)

    if dataset_name == "spiral":
        return fetch_spiral_dataset()

    from ucimlrepo import fetch_ucirepo

    if dataset_name == 'iris':
        dataset = fetch_ucirepo(id=53)
    elif dataset_name == 'tic-tac-toe':
        dataset = fetch_ucirepo(id=101)
    elif dataset_name == 'car evaluation':
        dataset = fetch_ucirepo(id=19)
    else:
        raise NotImplementedError(f"Dataset '{dataset_name}' is not implemented.")

    return dataset

def summarize_dataset(X, y):
    print("\nFeature Matrix (X):")
    summarize_vector(X)

    print("\nTarget Vector (y):")
    summarize_vector(y)

def summarize_vector(y):
    if isinstance(y, pd.Series):
        print(f"Shape: {y.shape}")
        if y.dtype.kind in 'biufc':  # Numeric types
            print(f"{'Min':<10}{'Max':<10}{'Mean':<10}")
            print("-" * 30)
            print(f"{y.min():<10.4f}{y.max():<10.4f}{y.mean():<10.4f}")
        else:
            print(f"Non-numeric column with {y.value_counts()} unique values:")
    elif isinstance(y, pd.DataFrame):
        print(f"Shape: {y.shape}")
        print(f"{'Column':<21}{'Min':<10}{'Max':<10}{'Mean':<10}{'Unique':<10}")
        print("-" * 56)
        for col in y.columns:
            if y[col].dtype.kind in 'biufc':  # Numeric types
                print(f"{col:<21}{y[col].min():<10.4f}{y[col].max():<10.4f}{y[col].mean():<10.4f}{'-':<10}")
            else:
                unique_count = y[col].nunique()
                print(f"{col:<21}{'N/A':<10}{'N/A':<10}{'N/A':<10}{unique_count:<10}")


def process_data(initial_state, target_state, data_split):


    # Split the encoded tensors into training, validation, and testing sets
    dataset_size = len(initial_state)
    val_size = int(data_split[1] * dataset_size)
    test_size = int(data_split[2] * dataset_size)

    train_size = dataset_size - test_size - val_size

    # Combine initial and target states into a single dataset for splitting
    full_dataset = torch.utils.data.TensorDataset(initial_state, target_state)

    # Perform the split
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42)
    )

    print(f"Data split complete:")
    print(f"Training set: {len(train_dataset)} samples")
    print(f"Validation set: {len(val_dataset)} samples")
    print(f"Testing set: {len(test_dataset)} samples")

    # Access training, validation, and testing data as needed
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1024, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=512, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=512, shuffle=False)

    return train_loader, val_loader, test_loader