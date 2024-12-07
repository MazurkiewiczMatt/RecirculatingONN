import numpy as np
import matplotlib.pyplot as plt

def plot_loss(training_loss, val_loss, val_period):
    epochs = np.arange(len(training_loss))
    epochs_val = np.arange(len(val_loss))*val_period

    plt.plot(epochs, training_loss, label="Training Loss", color="b")
    plt.plot(epochs_val, val_loss, label="Validation Loss", color="r")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()
