import torch

from device import device


def test_accuracy(best_model, test_loader):
    test_correct = 0
    total_test = 0
    best_model.eval()
    with torch.no_grad():
        for img, val_target in test_loader:
            img = img.to(device)
            val_target = val_target.to(device)
            output = best_model(img)

            _, predicted = torch.max(output, 1)
            test_correct += (predicted == val_target).sum().item()
            total_test += val_target.size(0)

    print("test accuracy: {}".format(test_correct / total_test))