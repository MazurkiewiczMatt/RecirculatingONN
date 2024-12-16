import torch
from torch.optim import Adam
from torchinfo import summary
from tqdm import tqdm

from datasets import get_dataset, summarize_dataset, process_data
from encoding import cavity_encoding
from TCMT import Circuit
from troubleshooting import plot_loss

def run_training(params):
    """
    Run the training process using a parameter dictionary. Any missing parameters
    will be set to a default value.
    """

    # Extract parameters from the dictionary, with defaults provided if not found
    dataset_name = params.get("dataset_name", "car evaluation")
    encoding_scheme = params.get("encoding_scheme", "phase")
    data_split = params.get("data_split", (0.6, 0.2, 0.2))
    bias_cavities = params.get("bias_cavities", 1)
    decoding_scheme = params.get("decoding_scheme", "categorical")
    learning_rate = params.get("learning_rate", 1e-3)
    num_epochs = params.get("num_epochs", 200)
    val_period = params.get("val_period", 5)
    plot_period = params.get("plot_period", 20)

    # FETCH, ENCODE, AND PROCESS DATA:
    dataset = get_dataset(dataset_name)
    print(f"\nDataset {dataset_name} ({dataset.metadata['name']}) loaded.")
    print(f"Abstract: {dataset.metadata['abstract']}")
    X = dataset.data.features
    y = dataset.data.targets

    initial_states, _ = cavity_encoding(X, encoding_scheme)
    target_states, loss_f = cavity_encoding(y, decoding_scheme)
    train_loader, val_loader, test_loader = process_data(initial_states, target_states, data_split)

    # SET UP THE OPTICAL CIRCUIT:
    num_input_modes = train_loader.dataset[0][0].size(0)
    num_output_modes = train_loader.dataset[0][1].size(0)
    num_modes = max(num_input_modes + bias_cavities, num_output_modes)
    circuit = Circuit.new(num_modes, num_input_modes)

    print("\nModel summary")
    summary(circuit)

    # TRAINING LOOP:
    optimizer = Adam(circuit.parameters(), lr=learning_rate)
    train_losses = []
    val_losses = []

    print("\nTraining started.\n")

    for epoch in (pbar := tqdm(range(num_epochs), desc="Training Progress")):
        circuit.train()
        train_loss = 0.0

        # Training loop
        for initial_batch, target_batch in train_loader:
            optimizer.zero_grad()
            circuit.zero_grad()

            predictions = circuit(initial_batch)[-1]
            output_predictions = predictions[:, :num_output_modes]

            loss = loss_f(output_predictions, target_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)


        if epoch % val_period == 0:
            val_loss = 0.0
            # Validation loop every val_period epochs
            circuit.eval()
            with torch.no_grad():
                for initial_batch, target_batch in val_loader:
                    predictions = circuit(initial_batch)[-1]
                    output_predictions = predictions[:, :num_output_modes]

                    loss = loss_f(output_predictions, target_batch)
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            val_losses.append(val_loss)

        # Update progress bar
        pbar.set_postfix(train_loss=f"{train_loss:.3f}", val_loss=f"{val_loss:.3f}")
        pbar.refresh()

        if epoch % plot_period == 0 and epoch > 0:
            plot_loss(train_losses, val_losses, val_period)

    print("\nTraining completed.\n")

    # TESTING PHASE:
    circuit.eval()
    test_loss = 0.0
    with torch.no_grad():
        for initial_batch, target_batch in test_loader:
            predictions = circuit(initial_batch)[-1]
            output_predictions = predictions[:, :num_output_modes]

            loss = loss_f(output_predictions, target_batch)
            test_loss += loss.item()

    test_loss /= len(test_loader)
    print(f"\nTest Loss: {test_loss:.4f}")

    if "csv_file_name" in params and params["csv_file_name"] is not None:
        import pandas as pd
        # Pad val_losses and test_loss with zeros
        max_length = len(train_losses)
        padded_val_losses = val_losses + [0] * (max_length - len(val_losses))
        padded_test_loss = [test_loss] + [0] * (max_length - 1)
        results_df = pd.DataFrame({
            "Train Losses": train_losses,
            "Val Losses": padded_val_losses,
            "Test Loss": padded_test_loss
        })
        # Save the DataFrame to a separate CSV file for each run
        results_df.to_csv(params["csv_file_name"], index=False)
        print(f"Results saved to '{params["csv_file_name"]}'")

    return train_losses, val_losses, test_loss