import torch

from torch.utils.data import WeightedRandomSampler


def create_weighted_sampler(df):

    class_counts = df[
        "target"
    ].value_counts().sort_index().values

    class_weights = 1.0 / class_counts

    sample_weights = [
        class_weights[label]
        for label in df["target"]
    ]

    sample_weights = torch.DoubleTensor(
        sample_weights
    )

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    return sampler