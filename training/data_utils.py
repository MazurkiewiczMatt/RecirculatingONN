import pandas as pd

from datasets import get_dataset, summarize_dataset, process_data
from encoding import cavity_encoding

def load_and_preprocess_data(params):
    """
    Load and preprocess the dataset according to the given parameters.
    Returns:
        train_loader, val_loader, test_loader
        num_modes, num_input_modes, num_output_modes
        target_codec, loss_f
        categories
    """
    dataset_name = params.get("dataset_name", "car evaluation")
    encoding_scheme = params.get("encoding_scheme", "phase")
    decoding_scheme = params.get("decoding_scheme", "categorical")
    data_split = params.get("data_split", (0.6, 0.2, 0.2))

    # Load dataset
    dataset = get_dataset(dataset_name)
    print(f"\nDataset {dataset_name} ({dataset.metadata['name']}) loaded.")
    print(f"Abstract: {dataset.metadata['abstract']}")

    X = dataset.data.features
    y = dataset.data.targets

    # Encode data
    initial_states, initial_codec = cavity_encoding(X, encoding_scheme)
    target_states, target_codec = cavity_encoding(y, decoding_scheme)
    loss_f = target_codec.loss_function()

    # Split into train, val, test
    train_loader, val_loader, test_loader = process_data(initial_states, target_states, data_split)

    # Determine number of modes
    num_input_modes = train_loader.dataset[0][0].size(0)
    num_output_modes = train_loader.dataset[0][1].size(0)

    cat_col = list(target_codec.categories.keys())[0]
    categories = target_codec.categories[cat_col]

    return train_loader, val_loader, test_loader, num_input_modes, num_output_modes, target_codec, loss_f, categories

def extract_labels(decoded_preds, decoded_targets):
    """
    Extract label arrays from decoded predictions and targets (DataFrame or Series).
    """
    if isinstance(decoded_preds, pd.DataFrame):
        pred_labels = decoded_preds.iloc[:, 0].values
        true_labels = decoded_targets.iloc[:, 0].values
    else:
        pred_labels = decoded_preds.values
        true_labels = decoded_targets.values
    return pred_labels, true_labels

