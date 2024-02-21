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
import time
import os
from cldm.logger import ImageLogger
from torch.utils.data import DataLoader
from datasets.vitonhd import VitonHDDataset


cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import albumentations as A
from omegaconf import OmegaConf
from PIL import Image
from peft import LoraConfig, get_peft_model


save_memory = False
disable_verbosity()
if save_memory:
    enable_sliced_attention()

# Configs
resume_path = ".ckpt/epoch=1-step=8687_ft.ckpt"
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

for param in model.parameters():
    param.requires_grad = False

for param in model.model.diffusion_model.output_blocks.parameters():
    param.requires_grad = True

for name, param in model.named_parameters():
    print(f"{name}: {param.requires_grad}")

trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad == True)
print(trainable_count)
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
