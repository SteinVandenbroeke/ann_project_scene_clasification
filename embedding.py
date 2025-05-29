import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import os
from device import device


def plot_embedding(model, test_loader, test_dataset, path, name='tsne_test_embeddings.png'):
    # Extract features from penultimate layer
    features = []
    labels_list = []

    def hook_fn(module, input, output):
        features.append(output.detach().cpu())

    hook = model.avgpool.register_forward_hook(hook_fn)

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            labels_list.extend(labels.numpy())
            _ = model(imgs)

    hook.remove()

    # Flatten features
    features = torch.cat(features, dim=0).squeeze(-1).squeeze(-1).numpy()

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(features)

    # Plot
    tsne_result = np.array(tsne_result)
    labels_list = np.array(labels_list)
    plt.figure(figsize=(10, 8))
    for label in np.unique(labels_list):
        idxs = labels_list == label
        plt.scatter(tsne_result[idxs, 0], tsne_result[idxs, 1], label=test_dataset.classes[label], alpha=0.6)
    plt.legend()
    plt.title("t-SNE visualization of test embeddings")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(path, name))
    plt.show()
