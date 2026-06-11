import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):

    def __init__(
        self,
        alpha=1,
        gamma=2,
        smoothing=0.1
    ):

        super().__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing

    def forward(
        self,
        inputs,
        targets
    ):

        ce_loss = F.cross_entropy(
            inputs,
            targets,
            reduction='none',
            label_smoothing=self.smoothing
        )

        pt = torch.exp(-ce_loss)

        focal_loss = self.alpha * (
            (1 - pt) ** self.gamma
        ) * ce_loss

        return focal_loss.mean()