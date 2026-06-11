import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

def get_model(num_classes=50):

    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    # 🔥 IMPORTANT: unfreeze
    for param in model.parameters():
        param.requires_grad = True

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model