from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np

def get_dataloaders(batch_size=128):

    # 🔥 Slightly better augmentation (minimal change)
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

    # ❗ Separate test transform (IMPORTANT FIX)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

    dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform_train
    )

    targets = np.array(dataset.targets)
    class_counts = np.bincount(targets)

    weights = 1. / class_counts
    sample_weights = weights[targets]

    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler
    )

    # ❗ FIX: use transform_test here
    test_dataset = datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform_test
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size
    )

    return train_loader, test_loader