import copy
import os.path

import torch
import torch.nn as nn
import torchvision
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import torchvision.utils as vutils
from utils import Save
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.animation as animation
from IPython.display import HTML
from tqdm import tqdm

from helper_functions import plot_results

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)

torch.manual_seed(41)
np.random.seed(41)

def train_and_test(train_loader,val_loader, test_loader, model, criterion, optimizer, scheduler, num_epochs, path):
    train_losses = []
    val_losses = []
    val_accuracies = []
    training_accuracies = []

    best_model = (0, None)
    save = Save(path)
    start_epoch = 0
    checkpoint = save.resume_if_exists(model, optimizer, num_epochs)
    if checkpoint is not None:
        start_epoch = checkpoint['epoch']
        train_losses = checkpoint.get('train_losses', [])
        val_losses = checkpoint.get('val_losses', [])
        val_accuracies = checkpoint.get('val_accuracies', [])
        training_accuracies = checkpoint.get('training_accuracies', [])
        best_model = checkpoint.get('best_model', (0, None))

    for epoch in range(num_epochs - start_epoch):
        model.train()
        train_loss = 0
        val_loss = 0
        total_train_loss = 0
        total_train = 0
        training_correct = 0
        for img, target in tqdm(train_loader):
            img = img.to(device)
            target = target.to(device)

            output = model(img)

            loss = criterion(output, target)# TODO: Compute the loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss = loss.item()
            total_train_loss += train_loss
            total_train += target.size(0)

            _, predicted = torch.max(output, 1)
            training_correct += (predicted == target).sum().item()

        correct = 0
        total_val = 0
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            for img, val_target in val_loader:
                img = img.to(device)
                val_target = val_target.to(device)
                output = model(img)
                loss = criterion(output, val_target)

                val_loss += loss.item()

                _, predicted = torch.max(output, 1)
                correct += (predicted == val_target).sum().item()
                total_val += val_target.size(0)

        avg_train_loss = total_train_loss / total_train
        avg_val_loss = val_loss / total_val
        val_accuracy = correct / total_val
        training_accuracy = training_correct / total_train
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        training_accuracies.append(training_accuracy)

        if val_accuracy > best_model[0]:
            best_model = (val_accuracy, copy.deepcopy(model))
        scheduler.step(val_loss)

        if (start_epoch + epoch) % 50 == 0:
            save.checkpoint(start_epoch, epoch, model, optimizer, {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies,
            'training_accuracies': training_accuracies,
            'best_model': best_model})

        print(f"Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.4f}")


        print('epoch [{}/{}], train loss:{:.4f}, validation loss:{:.4f}'.format(epoch + 1, num_epochs, avg_train_loss, avg_val_loss))


    print(best_model)
    #calculate test accuracy

    print(f"save num of epochs {num_epochs} {start_epoch}")
    save.checkpoint(num_epochs, 0, model, optimizer, {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'training_accuracies': training_accuracies,
        'best_model': best_model})

    plot_results(train_losses, val_losses, training_accuracies, val_accuracies, path)

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'training_accuracies': training_accuracies,
        'best_model': best_model}

model_per_size = {
    64:None,
    128:None,
    256:None
}