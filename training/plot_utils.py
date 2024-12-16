from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
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


def plot_confusion(pred_labels, true_labels, categories):
    """
    Plot the confusion matrix.
    """
    cm = confusion_matrix(true_labels, pred_labels, labels=categories)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=categories)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()