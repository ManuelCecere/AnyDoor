import cv2
import einops
import numpy as np
import torch
import random
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from cldm.model import create_model, load_state_dict
from cldm.cldm import ControlLDM
from cldm.ddim_hacked import DDIMSampler
from cldm.hack import disable_verbosity, enable_sliced_attention
from datasets.data_utils import *
import hiddenlayer as hl
from collections import namedtuple
from functools import partial
import time
import os
from cldm.logger import ImageLogger
from torch.utils.data import DataLoader
from datasets.vitonhd import VitonHDDataset
from ldm.modules.attention import BasicTransformerBlock, SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import ResBlock


cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import albumentations as A
from omegaconf import OmegaConf
from PIL import Image
from peft import LoraConfig, get_peft_model

torch.autograd.set_detect_anomaly(True)


class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.W_a = torch.nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        self.W_b = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x):
        x = self.alpha * (x @ self.W_a @ self.W_b)
        return x


class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha)

    def forward(self, x):
        return self.linear(x) + self.lora(x)


save_memory = False
disable_verbosity()
if save_memory:
    enable_sliced_attention()

# Configs
resume_path = ".ckpt/epoch=1-step=8687.ckpt"
batch_size = 1
logger_freq = 1000
learning_rate = 1e-5
sd_locked = False
only_mid_control = False
n_gpus = 1
accumulate_grad_batches = 1


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model("./configs/anydoor.yaml").cpu()
model.load_state_dict(load_state_dict(resume_path, location="cpu"))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

for name, param in model.named_parameters():
    param.requires_grad = False

# for name, param in model.named_parameters():
#     if "model.diffusion_model.output_blocks" in name:
#         param.requires_grad = True


lora_r = 8
lora_alpha = 16
lora_dropout = 0.05

assign_lora = partial(LinearWithLoRA, rank=lora_r, alpha=lora_alpha)

for block in model.model.diffusion_model.output_blocks:
    for layer in block:
        # Some Linear layers where I applied LoRA. Both raise the Error.
        # if isinstance(layer, ResBlock):
        # unfreeze parameters of ResBlock
        # for name, param in layer.named_parameters():
        #     param.requires_grad = True
        # # Access the emb_layers which is a Sequential containing Linear layers
        # emb_layers = layer.emb_layers
        # for i, layer in enumerate(emb_layers):
        #     if isinstance(layer, torch.nn.Linear):
        #         # Assign LoRA or any other modifications to the Linear layer
        #         emb_layers[i] = assign_lora(layer)
        if isinstance(layer, SpatialTransformer):
            for name, param in layer.named_parameters():
                param.requires_grad = True
            # layer.proj_in = assign_lora(layer.proj_in)


trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad == True)
print("trainable parameters: ", trainable_count)


with open("model_parameters.txt", "w") as file:
    for name, param in model.named_parameters():
        file.write(f"{name}: {param.requires_grad}\n")

with open("lora_model.txt", "w") as file:
    print(model, file=file)

# Datasets
DConf = OmegaConf.load("./configs/datasets.yaml")
dataset = VitonHDDataset(**DConf.Train.VitonHD)


dataloader = DataLoader(dataset, num_workers=8, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(
    gpus=n_gpus,
    strategy="ddp",
    precision=16,
    accelerator="gpu",
    callbacks=[logger],
    progress_bar_refresh_rate=1,
    accumulate_grad_batches=accumulate_grad_batches,
)


# Train!
trainer.fit(model, dataloader)
