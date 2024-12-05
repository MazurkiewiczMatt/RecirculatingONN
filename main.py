from datasets import get_dataset, summarize_dataset, process_data
from encoding import cavity_encoding


if __name__ == "__main__":

    # SETTINGS:

    dataset_name = "iris"  # available: "iris" (150 datapoints, 4 features); "tic-tac-toe" (958, 9); "car evaluation" (1728, 6)
    encoding_scheme = "phase"  # available: "phase"
    data_split = (0.6, 0.2, 0.2)
    bias_cavities = 1  # number of additional cavities, not used in input or readout. Their initial state is an extra degree of freedom.

    # FETCH, ENCODE, AND PROCESS DATA:

    dataset = get_dataset(dataset_name)
    print(f"\nDataset {dataset_name} ({dataset.metadata['name']}) loaded.")
    print(f"Abstract: {dataset.metadata['abstract']}")
    X = dataset.data.features
    y = dataset.data.targets
    # summarize_dataset(X, y)  # uncomment for a summary of the loaded dataset
    initial_states = cavity_encoding(X, encoding_scheme)
    target_states = cavity_encoding(y, encoding_scheme)
    train_loader, val_loader, test_loader = process_data(initial_states, target_states, data_split)

    # SET UP THE OPTICAL CIRCUIT:

    for initial_batch, target_batch in train_loader:
        input_cavity_n = initial_batch.size(1)  # Second dimension represents columns
        output_cavity_n = target_batch.size(1)





