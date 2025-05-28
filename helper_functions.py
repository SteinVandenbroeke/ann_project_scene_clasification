import os

import torch
from matplotlib import pyplot as plt
from torch.utils.serialization.config import save


def one_hot_encoding(target, num_classes):
    # Create a zero tensor of shape [batch_size, num_classes]
    one_hot = torch.zeros(target.size(0), num_classes, device=target.device)

    # Scatter 1s at the correct indices
    one_hot.scatter_(1, target.unsqueeze(1), 1)

    return one_hot

def plot_results(train_losses, val_losses, train_accuracies, val_accuracies, path):
    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', marker='o')
    plt.plot(val_losses, label='Val Loss', marker='x')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy', marker='o')
    plt.plot(val_accuracies, label='Val Accuracy', marker='x')
    plt.title('Validation Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(path, 'loss_and_accuracy.png'))
    plt.show()