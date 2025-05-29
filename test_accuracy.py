import torch

from device import device
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

def test_accuracy_and_conf_matrix(best_model, test_loader, test_dataset, path):
    test_correct = 0
    total_test = 0
    all_preds = []
    all_targets = []
    best_model.eval()
    with torch.no_grad():
        for img, val_target in test_loader:
            img = img.to(device)
            val_target = val_target.to(device)
            output = best_model(img)

            _, predicted = torch.max(output, 1)
            test_correct += (predicted == val_target).sum().item()
            total_test += val_target.size(0)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(val_target.cpu().numpy())

    # Compute confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100
    class_names = test_dataset.classes
    fig, ax = plt.subplots(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=class_names)
    disp.plot(ax=ax, xticks_rotation=45, cmap="Blues", values_format=".2f")
    plt.title("Normalized Confusion Matrix (%) on Test Set")
    plt.tight_layout()
    plt.savefig(f"{path}/confusion_matrix.png")
    plt.show()


    print("test accuracy: {}".format(test_correct / total_test))