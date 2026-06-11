import torch
import numpy as np


def mixup_data(
    x,
    y,
    alpha=0.8
):

    lam = np.random.beta(
        alpha,
        alpha
    )

    batch_size = x.size(0)

    index = torch.randperm(
        batch_size
    ).to(x.device)

    mixed_x = (
        lam * x
        + (1 - lam) * x[index]
    )

    y_a = y

    y_b = y[index]

    return mixed_x, y_a, y_b, lam


def mixup_criterion(
    criterion,
    pred,
    y_a,
    y_b,
    lam
):

    return (
        lam * criterion(pred, y_a)
        + (1 - lam) * criterion(pred, y_b)
    )