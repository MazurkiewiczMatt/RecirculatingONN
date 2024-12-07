import torch
from torch.optim import Adam
from torchinfo import summary
from tqdm import tqdm

from datasets import get_dataset, summarize_dataset, process_data
from encoding import cavity_encoding
from TCMT import Circuit
from troubleshooting import plot_loss

if __name__ == "__main__":

    # SETTINGS:

    dataset_name = "car evaluation"  # available: "iris" (150 datapoints, 4 features); "tic-tac-toe" (958, 9); "car evaluation" (1728, 6)
    encoding_scheme = "phase"  # available: "phase"
    data_split = (0.6, 0.2, 0.2)
    bias_cavities = 1  # number of additional cavities, not used in input or readout. Their initial state is an extra degree of freedom.
    decoding_scheme = "categorical"  # available: "phase", "categorical"

    # FETCH, ENCODE, AND PROCESS DATA:

    dataset = get_dataset(dataset_name)
    print(f"\nDataset {dataset_name} ({dataset.metadata['name']}) loaded.")
    print(f"Abstract: {dataset.metadata['abstract']}")
    X = dataset.data.features
    y = dataset.data.targets
    # summarize_dataset(X, y)  # uncomment for a summary of the loaded dataset
    initial_states, _ = cavity_encoding(X, encoding_scheme)
    target_states, loss_f = cavity_encoding(y, decoding_scheme)
    train_loader, val_loader, test_loader = process_data(initial_states, target_states, data_split)

    # SET UP THE OPTICAL CIRCUIT:
    num_input_modes = train_loader.dataset[0][0].size(0)  # Number of input cavities
    num_output_modes = train_loader.dataset[0][1].size(0)  # Number of output cavities
    num_modes = max(num_input_modes + bias_cavities, num_output_modes)
    circuit = Circuit.default(num_modes, num_input_modes)

    print("\nModel summary")
    summary(circuit)

    # TRAINING LOOP:
    optimizer = Adam(circuit.parameters(), lr=1e-3)
    num_epochs = 200
    train_losses = []
    val_losses = []

    val_period = 5 # run validation every 5 epochs
    plot_period = 20 # plot losses every 20 epochs

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
            # Validation loop every val_period epochs
            circuit.eval()
            val_loss = 0.0
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
