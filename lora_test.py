import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import gradcheck

from ldm.modules.attention import BasicTransformerBlock
from ldm.modules.diffusionmodules.openaimodel import ResBlock
from functools import partial


class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        self.W_a = nn.Parameter(torch.randn(in_dim, rank, dtype=torch.float64) * 0.01)
        self.W_b = nn.Parameter(torch.randn(rank, out_dim, dtype=torch.float64) * 0.01)
        self.alpha = alpha

    def forward(self, x):
        return self.alpha * (x @ self.W_a @ self.W_b)


class LinearWithLoRA(nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha)

    def forward(self, x):
        return self.linear(x) + self.lora(x)


class SimpleNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.res_block = ResBlock(
            channels=1280,
            emb_channels=1280,
            dropout=0.1,
            use_conv=False,
            use_scale_shift_norm=False,
            dims=2,
            use_checkpoint=True,
            up=False,
            down=False,
        )

    def forward(self, x):
        x = self.res_block(x)
        return x


# Initialize the network (ensure all parts are in double precision if not already)
model = SimpleNetwork().double()


assign_lora = partial(LinearWithLoRA, rank=8, alpha=1)

for i, layer in enumerate(model.res_block.emb_layers):
    if isinstance(layer, nn.Linear):
        model.res_block.emb_layers[i] = assign_lora(layer)

with open("lora_model_simple.txt", "w") as file:
    print(model, file=file)


# Create a sample input tensor
input_tensor = torch.randn(2, 1280, dtype=torch.float64, requires_grad=True)


# Perform gradcheck
test = gradcheck(model, (input_tensor,), eps=1e-6, atol=1e-4)
print("Gradcheck passed:", test)
