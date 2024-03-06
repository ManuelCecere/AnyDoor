from functools import partial
import cv2
import einops
import numpy as np
import torch
import random
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from cldm.model import create_model, load_state_dict
from cldm.hack import disable_verbosity, enable_sliced_attention
from datasets.data_utils import *
from cldm.logger import ImageLogger
from torch.utils.data import DataLoader, SubsetRandomSampler
from datasets.vitonhd import VitonHDDataset, VitonHDDataset_agnostic
from ldm.modules.attention import BasicTransformerBlock, SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import ResBlock
from pytorch_lightning.callbacks import StochasticWeightAveraging, ModelCheckpoint
from pytorch_lightning.tuner.tuning import Tuner


cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import albumentations as A
from omegaconf import OmegaConf

seed_everything(42, workers=True)


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
batch_size = 16
# value found with ligthinig lr_finder
learning_rate = 7.58e-8
sd_locked = False
only_mid_control = False
n_gpus = 1
accumulate_grad_batches = 4
# Datasets, the Viton agnostic uses a mask agnostic wrt the garment
DConf = OmegaConf.load("./configs/datasets.yaml")
dataset_train = VitonHDDataset_agnostic(**DConf.Train.VitonHD)
dataset_val = VitonHDDataset_agnostic(**DConf.Test.VitonHDTest)
print("Train: ", len(dataset_train))
print("Val: ", len(dataset_val))


def create_fractional_sampler(dataset, fraction):
    num_samples = len(dataset)
    indices = list(range(num_samples))
    sampled_indices = indices[: int(num_samples * fraction)]
    return SubsetRandomSampler(sampled_indices)


# use these samplers if you want to reduce the size of the datasets, for test purposes, pass it as parameters to the loaders
# val_sampler = create_fractional_sampler(dataset_val, fraction=0.01)
# train_sampler = create_fractional_sampler(dataset_train, fraction=0.01)

dataloader_train = DataLoader(
    dataset_train,
    num_workers=8,
    batch_size=batch_size,
    shuffle=False,
)
dataloader_val = DataLoader(
    dataset_val,
    num_workers=8,
    batch_size=batch_size,
    shuffle=False,
)


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model("./configs/anydoor.yaml").cpu()
model.load_state_dict(load_state_dict(resume_path, location="cpu"))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

for name, param in model.named_parameters():
    param.requires_grad = False

lora_r = 128
lora_alpha = 64

assign_lora = partial(LinearWithLoRA, rank=lora_r, alpha=lora_alpha)

for block in model.model.diffusion_model.output_blocks:
    for layer in block:
        # Some Linear layers where I applied LoRA. Both raise the Error.
        if isinstance(layer, ResBlock):
            # Access the emb_layers which is a Sequential containing Linear layers
            emb_layers = layer.emb_layers
            for i, layer in enumerate(emb_layers):
                if isinstance(layer, torch.nn.Linear):
                    # Assign LoRA or any other modifications to the Linear layer
                    emb_layers[i] = assign_lora(layer)
        if isinstance(layer, SpatialTransformer):
            layer.proj_in = assign_lora(layer.proj_in)

trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad == True)
print("trainable parameters: ", trainable_count)


with open("model_parameters.txt", "w") as file:
    for name, param in model.named_parameters():
        file.write(f"{name}: {param.requires_grad}\n")

with open("lora_model.txt", "w") as file:
    print(model, file=file)

# our checkpoint callback. It start at the end of every validation.
checkpoint_callback = ModelCheckpoint(
    dirpath= "/home/ubuntu/volume240/AnyDoorLoRA/LoRA_ft",
    monitor="val/loss",
    verbose=True,
    filename="epoch{epoch:01d}-step{step:03d}-val_loss{val/loss:.4f}",
    auto_insert_metric_name=False,
)

swa_callback = StochasticWeightAveraging(swa_lrs=1e-7)


trainer = pl.Trainer(
    gpus=n_gpus,
    precision=16,
    accelerator="gpu",
    callbacks=[checkpoint_callback, swa_callback],
    progress_bar_refresh_rate=1,
    accumulate_grad_batches=accumulate_grad_batches,
    default_root_dir="./LoRA_ft/checkpoints",
    max_epochs=3,
    val_check_interval=180,
    profiler="simple",
)

# tuner = Tuner(trainer)
# # Run learning rate finder
# lr_finder = tuner.lr_find(
#     model=model,
#     train_dataloaders=dataloader_train,
#     val_dataloaders=dataloader_val,
#     early_stop_threshold=None,
# )
# # Results can be found in
# print(lr_finder.results)

# # Plot with
# fig = lr_finder.plot(suggest=True)
# fig.show()

# # Pick point based on plot, or get suggestion
# print(lr_finder.suggestion())

# Train!
trainer.fit(model, train_dataloaders=dataloader_train, val_dataloaders=dataloader_val)
