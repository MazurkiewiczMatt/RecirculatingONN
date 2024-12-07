import pandas as pd
import torch
from torch.utils.data import random_split

def get_dataset(dataset_name):
    # available: "iris" (150 datapoints, 4 features); "tic-tac-toe" (958, 9); "car evaluation" (1728, 6)

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
    test_size = int(data_split[1] * dataset_size)
    val_size = int(data_split[2] * dataset_size)
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
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=256, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False)

    return train_loader, val_loader, test_loader