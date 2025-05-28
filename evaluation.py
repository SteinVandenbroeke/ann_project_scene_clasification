import copy
import math
import os.path
import subprocess
import time

import torch
import torch.nn as nn
import torchvision
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import torchvision.utils as vutils

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.animation as animation
from IPython.display import HTML
from torchvision.models import ResNet18_Weights
from tqdm import tqdm

from Vanilla_supervised_learning import train_and_test
from embedding import plot_embedding
from fine_tuning import fine_tuning
from helper_functions import plot_results
from models.SimCLR import SimCLRModel
from models.SupConModel import SupConModel
from pre_training import pre_training
from device import device
from test_accuracy import test_accuracy

print(device)

torch.manual_seed(41)

#Try out different sized
model_per_size = {
    64:None,
    128:None,
    256:None
}
start = time.time()

for img_size in [64, 128, 256]:
    startl = time.time()

    training_transform = transforms.Compose([
        transforms.Resize((img_size,img_size)),
        transforms.RandomVerticalFlip(),
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

    # Wrap them in DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=60, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=60, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=60, shuffle=False, num_workers=4)

    num_classes = len(train_dataset.classes)

    # #TODO interesting default weights
    model = torchvision.models.resnet18(weights=None).to(device)

    model.fc = nn.Linear(model.fc.in_features, num_classes).to(device)

    images, labels = next(iter(test_loader))

    num_epochs = 60

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    criterion = torch.nn.CrossEntropyLoss()


    model_per_size[img_size] = train_and_test(train_loader, val_loader, test_loader, model, criterion, optimizer, scheduler, num_epochs, f"saves/Vanilla/conf0/size_{img_size}")

    end = time.time()
    print(f"time elapse: {end - startl}")
    plot_embedding(model_per_size[img_size]["best_model"][1], test_loader, test_dataset, f"saves/Vanilla/conf0/size_{img_size}")

startl = time.time()

fig, ax = plt.subplots()

sizes = ["64", "128", "256"]
counts = [value["best_model"][0] for key, value in model_per_size.items()]

ax.bar(sizes, counts)

ax.set_ylabel('Accuracy')
ax.set_title('Accuracy per size')
ax.legend(title='Size of image')

plt.show()

###impact of training augmentation
training_transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),  # Converts PIL Image to Tensor
    transforms.Normalize((0.5,), (0.5,))
])

basis_transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),  # Converts PIL Image to Tensor
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.ImageFolder('./data/train', transform=training_transform)
val_dataset = datasets.ImageFolder('./data/val', transform=basis_transform)
test_dataset = datasets.ImageFolder('./data/test', transform=basis_transform)

# Wrap them in DataLoaders
train_loader = DataLoader(train_dataset, batch_size=60, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=60, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=60, shuffle=False, num_workers=4)

num_classes = len(train_dataset.classes)

# #TODO interesting default weights
model = torchvision.models.resnet18(weights=None).to(device)

model.fc = nn.Linear(model.fc.in_features, num_classes).to(device)

num_epochs = 60

optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

criterion = torch.nn.CrossEntropyLoss()

result = train_and_test(train_loader, val_loader, test_loader, model, criterion, optimizer, scheduler, num_epochs, f"saves/Vanilla/conf0/no_image_augmentation")

plt.figure(figsize=(12, 5))

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(result["val_losses"], label='Val loss without augmentation', marker='o')
plt.plot(model_per_size[256]["val_losses"], label='Val loss with augmentation', marker='x')
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(result["val_accuracies"], label='Val Accuracy without augmentation', marker='o')
plt.plot(model_per_size[256]["val_accuracies"], label='Val Accuracy with augmentation', marker='x')
plt.title('Validation Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join("plot_results", 'with_without_augmentation.png'))
plt.show()


### impact of Scheduler
model = torchvision.models.resnet18(weights=None).to(device)

model.fc = nn.Linear(model.fc.in_features, num_classes).to(device)

num_epochs = 60

optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005)
scheduler = MultiStepLR(optimizer, milestones=[math.ceil(num_epochs*0.6), math.ceil(num_epochs*0.8)], gamma=0.1)

criterion = torch.nn.CrossEntropyLoss()

result = train_and_test(train_loader, val_loader, test_loader, model, criterion, optimizer, scheduler, num_epochs, f"saves/Vanilla/conf0/MultiStepLR_sheduler")

plt.figure(figsize=(12, 5))

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(result["val_losses"], label='Val loss with MultiStepLR', marker='o')
plt.plot(model_per_size[256]["val_losses"], label='Val loss with ReduceLROnPlateau', marker='x')
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(result["val_accuracies"], label='Val Accuracy with MultiStepLR', marker='o')
plt.plot(model_per_size[256]["val_accuracies"], label='Val Accuracy with ReduceLROnPlateau', marker='x')
plt.title('Validation Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join("plot_results", 'compare_schedulers.png'))
plt.show()

print("validation accuracy: {}".format(result["best_model"][0]))
test_accuracy(model_per_size[256]["best_model"][1], test_loader)

end = time.time()
print(f"time elapse: {end - startl}")
startl = time.time()

result = pre_training(SimCLRModel, "saves/SimCLRModel/conf0/pre_train", 200, 60)
fine_tuning(result["best_model"], "saves/SimCLRModel/conf0/fine_tuning/0", 45, 1200, 60)

end = time.time()
print(f"time elapse: {end - startl}")
startl = time.time()

result = pre_training(SimCLRModel, "saves/SimCLRModel/conf0/pre_train", 400, 60)
fine_tuning(result["best_model"], "saves/SimCLRModel/conf0/fine_tuning/1", 45, 1200, 60)

end = time.time()
print(f"time elapse: {end - startl}")
startl = time.time()

result = pre_training(SimCLRModel, "saves/SimCLRModel/conf1/pre_train", 200, 92)
fine_tuning(result["best_model"], "saves/SimCLRModel/conf1/fine_tuning/0", 45, 1000, 92)

end = time.time()
print(f"time elapse: {end - startl}")
startl = time.time()

result = pre_training(SimCLRModel, "saves/SimCLRModel/conf1/pre_train", 400, 92)
fine_tuning(result["best_model"], "saves/SimCLRModel/conf1/fine_tuning/1", 45, 1000, 92)

end = time.time()
print(f"time elapse: {end - startl}")
startl = time.time()

result = pre_training(SimCLRModel, "saves/SimCLRModel/conf1/pre_train", 800, 92)
fine_tuning(result["best_model"], "saves/SimCLRModel/conf1/fine_tuning/2", 45, 1000, 92)

end = time.time()
print(f"time elapse: {end - startl}")
startl = time.time()

result = pre_training(SimCLRModel, "saves/SimCLRModel/conf2/pre_train", 200, 128)
fine_tuning(result["best_model"], "saves/SimCLRModel/conf2/fine_tuning/0", 45, 1000, 128)

end = time.time()
print(f"time elapse: {end - startl}")
startl = time.time()

end = time.time()
print(f"time elapse: {end - startl}")
startl = time.time()

result = pre_training(SimCLRModel, "saves/SimCLRModel/conf2/pre_train", 800, 128)
fine_tuevaluationning(result["best_model"], "saves/SimCLRModel/conf2/fine_tuning/1", 45, 1000, 128)

end = time.time()
print(f"time elapse: {end - startl}")
startl = time.time()

result = pre_training(SupConModel, "saves/SupConModel/conf0/pre_train", 400, 60)
fine_tuning(result["best_model"], "saves/SupConModel/conf0/fine_tuning/0", 45, 1200, 60)

end = time.time()
print(f"time elapse: {end - startl}")
startl = time.time()

result = pre_training(SupConModel, "saves/SupConModel/conf0/pre_train", 800, 60)
fine_tuning(result["best_model"], "saves/SupConModel/conf0/fine_tuning/1", 45, 1200, 60)

end = time.time()
print(f"time elapse: {end - startl}")
startl = time.time()

result = pre_training(SupConModel, "saves/SupConModel/conf1/pre_train", 400, 92)
fine_tuning(result["best_model"], "saves/SupConModel/conf1/fine_tuning/0", 45, 1000, 92)

end = time.time()
print(f"time elapse: {end - startl}")
startl = time.time()

result = pre_training(SupConModel, "saves/SupConModel/conf1/pre_train", 800, 92)
fine_tuning(result["best_model"], "saves/SupConModel/conf1/fine_tuning/1", 45, 1000, 92)

end = time.time()
print(f"time elapse: {end - startl}")
startl = time.time()

result = pre_training(SupConModel, "saves/SupConModel/conf2/pre_train", 400, 128)
fine_tuning(result["best_model"], "saves/SupConModel/conf2/fine_tuning", 45, 1000, 128)
end = time.time()

print(f"time elapse: {end - startl}")

print(f"total time elapse: {end - start}")

start = time.time()
startl = time.time()

result = pre_training(SupConModel, "saves/SupConModel/conf3/pre_train", 400, 60, batch_size=512)
fine_tuning(result["best_model"], "saves/SupConModel/conf3/fine_tuning/0", 45, 1200, 60)

end = time.time()
print(f"time elapse: {end - startl}")
startl = time.time()

result = pre_training(SupConModel, "saves/SupConModel/conf3/pre_train", 800, 60, batch_size=512)
fine_tuning(result["best_model"], "saves/SupConModel/conf3/fine_tuning/1", 45, 1200, 60)

end = time.time()
print(f"time elapse: {end - startl}")
startl = time.time()
