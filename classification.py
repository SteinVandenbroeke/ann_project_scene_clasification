import os.path

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import torchvision.utils as vutils

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.animation as animation
from IPython.display import HTML

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torchvision.models.resnet18()

def supervised_learning():
    def train(model, train_loader, num_epochs=5):
        criterion =  torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(num_epochs):
            for data in train_loader:
                img, target = data
                img = img.to(device)

                output = model(img)
                loss =  criterion(output, target)# TODO: Compute the loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, loss.item()))

    model = torchvision.models.resnet18()

    print(os.path.abspath(os.getcwd()))
    train_dataset = datasets.ImageFolder('./15scene/train')
    val_dataset = datasets.ImageFolder('./15scene/validate')
    test_dataset = datasets.ImageFolder('./15scene/test')

    # Wrap them in DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    train(model, train_loader)

supervised_learning()