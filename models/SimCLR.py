#TODO
import torch
import torchvision
from torch import nn
import torch.nn.functional as F

class SimCLRModel(nn.Module):
    def __init__(self, num_classes, projection_dim=128):
        super(SimCLRModel, self).__init__()
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
    def info_nce_loss(self, feats):
        # Compute cosine similarity between all pairs of feature vectors
        cos_sim = F.cosine_similarity(feats[:, None, :], feats[None, :, :], dim=-1)

        # Create a mask for the diagonal elements (self-similarity), which should not be considered
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)

        # Mask out the diagonal (self-similarity) by setting those entries to a very low value
        cos_sim.masked_fill_(self_mask, -9e15)

        # Create a mask for positive pairs (shifting the self-mask by half of the batch size)
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)

        # Apply temperature scaling to the cosine similarities to control the softness of the distribution
        cos_sim = cos_sim / self.temperature

        # Calculate the negative log-likelihood (NLL) loss, as described in the original SimCLR paper
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)

        # Return the mean NLL loss, the cosine similarity matrix, and the positive mask
        return nll.mean(), cos_sim, pos_mask


