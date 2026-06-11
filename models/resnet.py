import torch.nn as nn
import torchvision.models as models

def get_model(num_classes=10):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Replace final layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model