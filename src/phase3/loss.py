import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, inputs, targets):
        targets = targets.to(inputs.device)

        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)

        loss = ((1 - pt) ** self.gamma) * ce_loss
        return loss.mean()


class ClassBalancedLoss(nn.Module):
    def __init__(self, samples_per_cls, beta=0.9999):
        super().__init__()

        # 🔥 FIX 1: Safe tensor conversion (no warning)
        if not isinstance(samples_per_cls, torch.Tensor):
            samples_per_cls = torch.tensor(samples_per_cls, dtype=torch.float32)
        else:
            samples_per_cls = samples_per_cls.clone().detach().float()

        effective_num = 1.0 - torch.pow(beta, samples_per_cls)
        weights = (1.0 - beta) / (effective_num + 1e-8)

        # 🔥 Normalize properly
        weights = weights / weights.sum() * len(weights)

        # 🔥 register buffer (good practice)
        self.register_buffer("weights", weights)

    def forward(self, inputs, targets):
        targets = targets.to(inputs.device)

        # 🔥 FIX 2: move weights to same device
        weights = self.weights.to(inputs.device)

        return F.cross_entropy(inputs, targets, weight=weights)