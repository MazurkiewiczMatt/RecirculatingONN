import torch
from torch.optim import Adam


from tqdm import tqdm
import pandas as pd

from .data_utils import load_and_preprocess_data, extract_labels
from .model_utils import initialize_circuit
from .plot_utils import plot_confusion, plot_loss


def save_model(circuit, model_path):
    """
    Save the model's state dictionary to the specified path.
    """
    torch.save(circuit.state_dict(), model_path)
    print(f"Model saved to '{model_path}'")


def load_model(circuit, model_path, device='cpu'):
    """
    Load the model's state dictionary from the specified path.
    """
    circuit.load_state_dict(torch.load(model_path, map_location=device))
    circuit.to(device)
    print(f"Model loaded from '{model_path}'")
    return circuit


def train_one_epoch(circuit, train_loader, optimizer, loss_f, num_output_modes):
    """
    Train the model for one epoch and return the average training loss.
    """
    circuit.train()
    train_loss = 0.0
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
    return train_loss


def validate_model(circuit, val_loader, loss_f, num_output_modes):
    """
    Validate the model and return the average validation loss.
    """
    circuit.eval()
    val_loss = 0.0
    with torch.no_grad():
        for initial_batch, target_batch in val_loader:
            predictions = circuit(initial_batch)[-1]
            output_predictions = predictions[:, :num_output_modes]

            loss = loss_f(output_predictions, target_batch)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    return val_loss


def train_model(circuit, train_loader, val_loader, loss_f, params):
    """
    Train the model for the specified number of epochs and return the training and validation losses.
    """
    learning_rate = params.get("learning_rate", 1e-3)
    num_epochs = params.get("num_epochs", 200)
    val_period = params.get("val_period", 5)
    plot_period = params.get("plot_period", 20)

    num_output_modes = train_loader.dataset[0][1].size(0)
    optimizer = Adam(circuit.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []

    print("\nTraining started.\n")

    for epoch in (pbar := tqdm(range(num_epochs), desc="Training Progress")):

        if "cavityless_pretraining" in params and params["cavityless_pretraining"] is not None:
            if epoch == params["cavityless_pretraining"]:
                circuit.enable_cavity_training()
                optimizer.add_param_group({'params': circuit.kappa})
                optimizer.add_param_group({'params': circuit.omega})
                optimizer.add_param_group({'params': circuit.nonlinearity})
        train_loss = train_one_epoch(circuit, train_loader, optimizer, loss_f, num_output_modes)
        train_losses.append(train_loss)

        if epoch % val_period == 0:
            val_loss = validate_model(circuit, val_loader, loss_f, num_output_modes)
            val_losses.append(val_loss)

        # Update progress bar
        if val_loss is not None:
            pbar.set_postfix(train_loss=f"{train_loss:.3f}", val_loss=f"{val_loss:.3f}")
        else:
            pbar.set_postfix(train_loss=f"{train_loss:.3f}")
        pbar.refresh()

        # Plot losses periodically
        if (epoch+1) % plot_period == 0:
            plot_loss(train_losses, val_losses, val_period)

    print("\nTraining completed.\n")
    return train_losses, val_losses


def test_model(circuit, test_loader, loss_f, target_codec):
    """
    Test the model, returning the test loss, predictions, and targets.
    """
    circuit.eval()
    test_loss = 0.0
    all_preds = []
    all_targets_list = []

    num_output_modes = test_loader.dataset[0][1].size(0)
    with torch.no_grad():
        for initial_batch, target_batch in test_loader:
            predictions = circuit(initial_batch)[-1]
            output_predictions = predictions[:, :num_output_modes]

            # Compute test loss
            loss = loss_f(output_predictions, target_batch)
            test_loss += loss.item()

            # Convert complex output to amplitude squared
            amplitude_squared = torch.abs(output_predictions) ** 2
            pred_idxs = torch.argmax(amplitude_squared, dim=1)

            # Convert indices to one-hot
            preds_one_hot = torch.zeros_like(output_predictions, dtype=torch.float32)
            preds_one_hot.scatter_(1, pred_idxs.unsqueeze(1), 1.0)

            all_preds.append(preds_one_hot)
            all_targets_list.append(target_batch)

    test_loss /= len(test_loader)

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets_list, dim=0)

    decoded_preds = target_codec.decode(all_preds)
    decoded_targets = target_codec.decode(all_targets)

    return test_loss, decoded_preds, decoded_targets


def save_results(train_losses, val_losses, test_loss, params):
    """
    Save the training, validation, and test results to a CSV file if requested.
    """
    if "csv_file_name" in params and params["csv_file_name"] is not None:
        # Pad val_losses and test_loss with zeros if lengths differ
        max_length = len(train_losses)
        padded_val_losses = val_losses + [0] * (max_length - len(val_losses))
        padded_test_loss = [test_loss] + [0] * (max_length - 1)
        results_df = pd.DataFrame({
            "Train Losses": train_losses,
            "Val Losses": padded_val_losses,
            "Test Loss": padded_test_loss
        })
        results_df.to_csv(params["csv_file_name"], index=False)
        print(f"Results saved to '{params['csv_file_name']}'")


def run_training(params):
    """
    Run the entire training process using the given parameters.
    """
    # Fetch and process data
    (train_loader, val_loader, test_loader,
     num_input_modes, num_output_modes,
     target_codec, loss_f, categories) = load_and_preprocess_data(params)

    # Set up the circuit
    params["num_input_modes"] = num_input_modes
    params["num_output_modes"] = num_output_modes
    circuit = initialize_circuit(params)

    # If loading a pre-trained model
    if "model_load_path" in params and params["model_load_path"] is not None:
        device = params.get("device", "cpu")
        circuit = load_model(circuit, params["model_load_path"], device=device)

    # Train the model
    train_losses, val_losses = train_model(circuit, train_loader, val_loader, loss_f, params)

    # Test the model
    test_loss, decoded_preds, decoded_targets = test_model(circuit, test_loader, loss_f, target_codec)
    print(f"\nTest Loss: {test_loss:.4f}")

    # Extract labels
    pred_labels, true_labels = extract_labels(decoded_preds, decoded_targets)

    # Plot confusion matrix
    plot_confusion(pred_labels, true_labels, categories)

    # Save results if required
    save_results(train_losses, val_losses, test_loss, params)

    # Save model if required
    if "model_file_name" in params and params["model_file_name"] is not None:
        save_model(circuit, params["model_file_name"])

    return train_losses, val_losses, test_loss
