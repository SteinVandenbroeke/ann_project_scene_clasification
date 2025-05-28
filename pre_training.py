import copy
import os.path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm
from device import device
from models.SimCLR import SimCLRModel
from models.SupConModel import SupConModel
from utils import Transformations, Save

torch.manual_seed(41)
np.random.seed(41)

def pre_training(model_type: SimCLRModel or SupConModel, path, num_epochs, img_crop_size=60, batch_size=256, learning_rate=0.001):
    train_dataset = datasets.ImageFolder('./data/train', transform=Transformations(img_crop_size, model_type))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    num_classes = len(train_dataset.classes)

    #
    # # Initialize model and optimizer
    model = model_type(num_classes).to(device)
    model.pre_training = True
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    save = Save(path)
    start_epoch = 0
    top1_list = []
    top5_list = []
    mean_pos_list = []
    best_model = (0, None)
    checkpoint = save.resume_if_exists(model, optimizer, num_epochs)
    if checkpoint is not None:
        start_epoch = checkpoint['epoch']
        top1_list = checkpoint.get('top1_list', [])
        top5_list = checkpoint.get('top5_list', [])
        mean_pos_list = checkpoint.get('mean_pos_list', [])
        best_model = checkpoint.get('best_model', (0, None))

    model.to(device)
    model.train()

    total_train_loss = 0
    total_train = 0

    print(f"num of epochs {num_epochs} {start_epoch}")
    # Pretrain with contrastive learning
    epoch = 0
    for epoch in range(num_epochs - start_epoch):
        model.train()
        total_loss = 0
        top1_acc = 0
        top5_acc = 0
        mean_pos = 0
        avg_loss = 0
        num_samples = len(train_loader.dataset) * 2
        for imgs, labels in tqdm(train_loader):
            labels = labels.to(device)

            if model_type == SimCLRModel:
                imgs = torch.cat(imgs, dim=0).to(device)
                out = model(imgs)
                loss , cos_sim, pos_mask = model.info_nce_loss(out)
                # Ranking metrics
                comb_sim = torch.cat([
                    cos_sim[pos_mask][:, None],
                    cos_sim.masked_fill(pos_mask, -9e15)
                ], dim=-1)
                sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)

            else:
                out = model(imgs.to(device))
                loss, cos_sim, pos_mask = model.info_nce_loss(out, labels)
                positive_scores = []
                for i in range(cos_sim.size(0)):
                    positive_scores.append(cos_sim[i][pos_mask[i]])
                loss , cos_sim, pos_mask = model.info_nce_loss(out, labels)
                negative_scores = cos_sim.masked_fill(pos_mask, -9e15)
                positive_scores_padded = pad_sequence(positive_scores, batch_first=True, padding_value=-9e15)
                comb_sim = torch.cat([positive_scores_padded, negative_scores], dim=1)
                sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_train_loss += total_loss
            total_train += labels.size(0)

            avg_loss = total_loss / num_samples
            top1_acc += (sim_argsort == 0).float().sum().item()
            top5_acc += (sim_argsort < 5).float().sum().item()
            mean_pos += (1 + sim_argsort.float()).sum().item()
            total_loss += loss.item() * imgs.size(0)

            if top5_acc > best_model[0]:
                best_model = (top5_acc, copy.deepcopy(model))

        #scheduler.step(total_loss)
        # because of two views
        print(f"Epoch [{start_epoch + epoch + 1}/{num_epochs}] Loss: {total_loss / num_samples:.4f}, "
              f"Top1 Acc: {top1_acc / num_samples:.4f}, "
              f"Top5 Acc: {top5_acc / num_samples:.4f}, "
              f"Mean Pos: {mean_pos / num_samples:.2f}, "
              f"LR: {scheduler.get_last_lr()}")

        avg_loss = total_loss / num_samples
        avg_top1 = top1_acc / num_samples
        avg_top5 = top5_acc / num_samples
        avg_pos = mean_pos / num_samples

        top1_list.append(avg_top1)
        top5_list.append(avg_top5)
        mean_pos_list.append(avg_pos)

        if (start_epoch + epoch) % 50 == 0:
            save.checkpoint(start_epoch, epoch, model, optimizer, {'loss': avg_loss,
            'top1_list': top1_list,
            'top5_list': top5_list,
            'mean_pos_list': mean_pos_list,
            'best_model': best_model})

    epochs = range(1, len(top1_list) + 1)
    plt.figure()
    plt.plot(epochs, top1_list, label='Top-1 Accuracy', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Top-1 Accuracy')
    plt.title('SimCLR Top-1 Accuracy')
    plt.grid(True)
    plt.savefig(os.path.join(save.save_path, 'top1_accuracy.png'))
    plt.show()

    plt.figure()
    plt.plot(epochs, top5_list, label='Top-5 Accuracy', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Top-5 Accuracy')
    plt.title('SimCLR Top-5 Accuracy')
    plt.grid(True)
    plt.savefig(os.path.join(save.save_path, 'top5_accuracy.png'))
    plt.show()

    plt.figure()
    plt.plot(epochs, mean_pos_list, label='Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('SimCLR Training Loss')
    plt.grid(True)
    plt.savefig(os.path.join(save.save_path, 'training_loss.png'))
    plt.show()

    save.checkpoint(num_epochs, 0, model, optimizer, {
                                                           'top1_list': top1_list,
                                                           'top5_list': top5_list,
                                                           'mean_pos_list': mean_pos_list,
                                                            'best_model': best_model})

    print("Plots saved in:", save.save_path)
    return {
        'best_model': best_model[1],
        'model': model,
           'top1_list': top1_list,
           'top5_list': top5_list,
           'mean_pos_list': mean_pos_list}