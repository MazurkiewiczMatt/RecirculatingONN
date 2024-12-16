from training import run_training


if __name__ == "__main__":

    params = {
        "dataset_name": "iris",                     # Available: "iris", "tic-tac-toe", "car evaluation"
        "encoding_scheme": "phase",                 # Available: "phase"
        "data_split": (0.6, 0.2, 0.2),
        "decoding_scheme": "categorical",           # Available: "phase", "categorical"
        "bias_cavities": 1,
        "learning_rate": 1e-3,                      # Parameter for optimizer
        "num_epochs": 2,                         # Number of epochs for training
        "val_period": 5,                            # Calculate validation loss every X epochs
        "plot_period": 50,                          # Plot training and val loss every X epochs
        "csv_file_name": "losses_test_run.csv",     # Save the array of losses to the file
    }

    train_losses, val_losses, test_loss = run_training(params)


