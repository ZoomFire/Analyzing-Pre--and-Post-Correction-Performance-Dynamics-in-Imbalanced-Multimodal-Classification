import torch

from model import PaSSTModel

model = PaSSTModel()

dummy = torch.randn(2, 1, 128, 501)

output = model(dummy)

print(output.shape)