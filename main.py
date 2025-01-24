from training import run_training


if __name__ == "__main__":



    for i in range(20):
        for is_temporal_DOF in [True, False]:
            params = {
                "dataset_name": "spiral",                     # Available: "iris", "tic-tac-toe", "car evaluation"
                "encoding_scheme": "phase",                 # Available: "phase"
                "data_split": (0.8, 0.1, 0.1),
                "decoding_scheme": "categorical",           # Available: "phase", "categorical"
                "bias_cavities": 2,
                "learning_rate": 1e-3,                      # Parameter for optimizer
                "num_epochs": 2000,                            # Number of epochs for training
                "val_period": 5,                            # Calculate validation loss every X epochs
                "plot_period": 50,                          # Plot training and val loss every X epochs
                "csv_file_name": f"temporal_DOF/run_{is_temporal_DOF}_{i+100}.csv",     # Save the array of losses to the file
                "model_file_name": f"temporal_DOF/run_{is_temporal_DOF}_{i+100}",          # save trained model here
                "learnable_omega": True,                    # are frequencies of resonant cavities learnable?
                "learnable_nonlinearity": True,             # are kerr effect coefficients learnable?
                "learnable_coupling": True,                 # are coupling coefficients learnable?
                "learnable_interferometer": True,
                "computation_time": 2.0,
                "regularization": 1e-6,
                "initialization_omega": None,
                "initialization_nonlinearity": 0.3,
                "initialization_coupling": None,
                "cavityless_pretraining": None,
                "temporal_DOF": is_temporal_DOF,
            }

            train_losses, val_losses, test_loss = run_training(params)

