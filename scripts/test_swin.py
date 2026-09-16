import torch

from torchvision.models import swin_t
from torchvision.models import Swin_T_Weights


print("Loading Swin Transformer...")


weights = Swin_T_Weights.DEFAULT

model = swin_t(
    weights=weights
)


print("Swin Transformer loaded successfully.")


print()
print("Model parameters:")

total_parameters = sum(
    p.numel()
    for p in model.parameters()
)

print(total_parameters)


print()
print("Testing one image...")


x = torch.randn(
    1,
    3,
    224,
    224
)


with torch.no_grad():

    output = model(x)


print("Output shape:")

print(output.shape)


print()
print("Test successful.")