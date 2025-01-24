from training import run_training


if __name__ == "__main__":



    for i in range(20):
        for is_nonlinear, is_cavity_trainable, is_U_variable in [[True, True, True], [False, True, True]]:
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
                "csv_file_name": f"fixed_results/nonlinearityfinal_{is_nonlinear}_{is_cavity_trainable}_{is_U_variable}_{i+100}.csv",     # Save the array of losses to the file
                "model_file_name": f"fixed_results/nonlinearityfinal_{is_nonlinear}_{is_cavity_trainable}_{is_U_variable}_{i+100}",          # save trained model here
                "learnable_omega": is_cavity_trainable,                    # are frequencies of resonant cavities learnable?
                "learnable_nonlinearity": (is_cavity_trainable and is_nonlinear),             # are kerr effect coefficients learnable?
                "learnable_coupling": is_cavity_trainable,                 # are coupling coefficients learnable?
                "learnable_interferometer": is_U_variable,
                "computation_time": 2.0,
                "regularization": 1e-6,
                "initialization_omega": None,
                "initialization_nonlinearity": 0.0 if not(is_nonlinear) else 0.3,
                "initialization_coupling": None,
                "cavityless_pretraining": None,
            }

            train_losses, val_losses, test_loss = run_training(params)

