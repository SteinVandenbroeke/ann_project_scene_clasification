#TODO
import torch
import torchvision
from torch import nn
import torch.nn.functional as F

class SupConModel(nn.Module):
    def __init__(self, num_classes, projection_dim=128):
        super(SupConModel, self).__init__()
        self.encoder = torchvision.models.resnet18(weights=None)
        in_dim = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()
        self.projector = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )
        self.linear_classifier = nn.Linear(in_dim, num_classes)
        self.encoder.fc.in_features = in_dim
        self.temperature = 0.5
        self.avgpool = self.encoder.avgpool
        self.pre_training = False

    def forward(self, x):
        if self.pre_training:
            h = self.encoder(x)
            z = self.projector(h)
            return z

        with torch.no_grad():
            h = self.encoder(x)
        z = self.linear_classifier(h)
        return z

    #from https://colab.research.google.com/github/phlippe/uvadlc_notebooks/blob/master/docs/tutorial_notebooks/tutorial17/SimCLR.ipynb#scrollTo=KSrup4nQq49j
    def info_nce_loss(self, feats, labels):
        cos_sim = torch.matmul(feats, feats.T)
        mask = torch.eye(cos_sim.size(0), device=cos_sim.device).bool()
        cos_sim.masked_fill_(mask, -9e15)

        label_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        label_mask.fill_diagonal_(False)  # remove self-similarity

        cos_sim = cos_sim / self.temperature
        cos_sim = cos_sim - torch.max(cos_sim, dim=1, keepdim=True)[0]  # stability

        exp_cos_sim = torch.exp(cos_sim)
        pos_mask = label_mask

        pos_sum = exp_cos_sim * pos_mask
        all_sum = exp_cos_sim.sum(dim=1)

        eps = 1e-8
        loss = -torch.log(pos_sum.sum(dim=1) / (all_sum + eps) + eps)
        loss = loss.mean()

        return loss, cos_sim, pos_mask