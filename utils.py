import os

import torch
import torchvision
from matplotlib import pyplot as plt
from torchvision import transforms
from device import device
from models.SimCLR import SimCLRModel
from models.SupConModel import SupConModel


class Save():
    def __init__(self, save_path):
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

    def resume_if_exists(self, model, optimizer, epoch):
        saved_epochs = [int(f.split('_epoch')[-1].split('.pth')[0])
                        for f in os.listdir(self.save_path) if f.startswith('simclr_epoch') and int(f.split('_epoch')[-1].split('.pth')[0]) <= epoch]
        if saved_epochs:
            latest_epoch = max(saved_epochs)
            print(os.path.join(self.save_path, f"simclr_epoch{latest_epoch}.pth"))
            latest_checkpoint = os.path.join(self.save_path, f"simclr_epoch{latest_epoch}.pth")
            return self.load_checkpoint(model, optimizer, latest_checkpoint, device)
        else:
            return None

    def load_checkpoint(self, model, optimizer, checkpoint_path, device):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']

        print(f"Loaded file {checkpoint_path} (Epoch {start_epoch})")
        return checkpoint

    def checkpoint(self, start_epoch, epoch, model, optimizer, extra: dict = None):
        checkpoint_path = os.path.join(self.save_path, f"simclr_epoch{start_epoch + epoch}.pth")
        torch.save({
            'epoch': start_epoch + epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        } | extra, checkpoint_path)
        print(f"Model saved to {checkpoint_path}")

class Transformations(object):
    def __init__(self, crop_size=60, model_type: (SimCLRModel or SupConModel) = SimCLRModel):
        self.transformer = transforms.Compose([
                                transforms.RandomResizedCrop(crop_size),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
                                transforms.RandomGrayscale(p=0.2),
                                transforms.Resize(crop_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))
                            ])
        self.model_type = model_type

    def __call__(self, x):
        if self.model_type == SupConModel:
            return self.transformer(x)
        return [self.transformer(x), self.transformer(x)]

def show_images(dataset, num_images = 6):
    imgs = torch.stack([img for idx in range(num_images) for img in dataset[idx][0]], dim=0)
    img_grid = torchvision.utils.make_grid(imgs, nrow=6, normalize=True, pad_value=0.9)
    img_grid = img_grid.permute(1, 2, 0)

    plt.figure(figsize=(10, 5))
    plt.title('Augmented image examples of the STL10 dataset')
    plt.imshow(img_grid)
    plt.axis('off')
    plt.show()
    plt.close()