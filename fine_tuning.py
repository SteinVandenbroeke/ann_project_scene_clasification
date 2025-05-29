import copy
import os.path
import time

import torch
import torch.nn as nn
import torchvision
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
from torchvision.models import ResNet18_Weights
from tqdm import tqdm
from PIL.ImageFilter import GaussianBlur
from torch.distributed import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import torchvision.utils as vutils
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.animation as animation
from IPython.display import HTML

from embedding import plot_embedding
from helper_functions import plot_results
from device import device
from test_accuracy import test_accuracy_and_conf_matrix
from utils import Save


def fine_tuning(model, path, num_epochs, batch_size=1000, img_size=256):
    training_transform = transforms.Compose([
        transforms.Resize((img_size,img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # Converts PIL Image to Tensor
        transforms.Normalize((0.5,), (0.5,))
    ])

    basis_transform = transforms.Compose([
        transforms.Resize((img_size,img_size)),
        transforms.ToTensor(),  # Converts PIL Image to Tensor
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.ImageFolder('./data/train', transform=training_transform)
    val_dataset = datasets.ImageFolder('./data/val', transform=basis_transform)
    test_dataset = datasets.ImageFolder('./data/test', transform=basis_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    num_classes = len(train_dataset.classes)

    test_dataset = datasets.ImageFolder('./data/test', transform=basis_transform)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=4)
    plot_embedding(model, test_loader, test_dataset, path, 'tsne_test_embeddings_before_linear_layer.png')

    model.pre_training = False
    for param in model.encoder.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(model.linear_classifier.parameters(), lr=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    criterion = nn.CrossEntropyLoss()

    start_epoch = 0
    train_losses = []
    val_losses = []
    val_accuracies = []
    training_accuracies = []
    best_model = (0, None)
    save = Save(path)
    checkpoint = save.resume_if_exists(model, optimizer, num_epochs)
    if checkpoint is not None:
        start_epoch = checkpoint['epoch']
        train_losses = checkpoint.get('train_losses', [])
        val_losses = checkpoint.get('val_losses', [])
        val_accuracies = checkpoint.get('val_accuracies', [])
        training_accuracies = checkpoint.get('training_accuracies', [])
        best_model = checkpoint.get('best_model', (0, None))

    for epoch in range(num_epochs - start_epoch):
        model.linear_classifier.train()
        epoch_train_loss = 0
        total = 0
        correct = 0
        training_correct = 0
        for img, labels in train_loader:
            img, labels = img.to(device), labels.to(device)

            outputs = model(img)

            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item() * labels.size(0)
            total += labels.size(0)

            _, predicted = torch.max(outputs, 1)
            training_correct += (predicted == labels).sum().item()

        avg_train_loss = epoch_train_loss / total

        # Validation
        model.linear_classifier.eval()
        val_loss = 0
        correct = 0
        total_val = 0

        with torch.no_grad():
            for img, val_target in val_loader:
                img, val_target = img.to(device), val_target.to(device)
                output = model(img)
                loss = criterion(output, val_target)
                val_loss += loss.item() * val_target.size(0)
                _, predicted = torch.max(output, 1)
                correct += (predicted == val_target).sum().item()
                total_val += val_target.size(0)

        avg_val_loss = val_loss / total_val
        val_accuracy = correct / total_val
        training_accuracy = training_correct / total
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        training_accuracies.append(training_accuracy)
        scheduler.step(val_loss)

        if val_accuracy > best_model[0]:
            best_model = (val_accuracy, copy.deepcopy(model))

        if (start_epoch + epoch) % 50 == 0:
            save.checkpoint(start_epoch, epoch, model, optimizer, {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies,
            'training_accuracies': training_accuracies,
            'best_model': best_model})

        print(f"Linear Probe Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Train acc {training_accuracy * 100}, Val Acc: {val_accuracy*100:.2f}%, LR: ({scheduler.get_last_lr()})")

    print(val_accuracies)

    print(f"save num of epochs {num_epochs} {start_epoch}")
    save.checkpoint(0, num_epochs, model, optimizer, {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'training_accuracies': training_accuracies,
        'best_model': best_model})

    plot_results(train_losses, val_losses, training_accuracies, val_accuracies, path)
    print("validation accuracy: {}".format(best_model[0]))
    test_accuracy_and_conf_matrix(best_model[1], test_loader, test_dataset, path)
    plot_embedding(best_model[1], test_loader, test_dataset, path)
    return best_model[1]
