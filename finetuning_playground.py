import cv2
import einops
import numpy as np
import torch
import random
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from PeftModelLightning import PeftModelLightning
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
resume_path = ".ckpt/epoch=1-step=8687_train.ckpt"
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


target_mods = []
for name, module in model.named_modules():
    # print(f"{name:20} : {type(module).__name__}")
    if (
        type(module).__name__ == "Conv2d"
        and "model.diffusion_model.input_blocks" in name
    ):
        target_mods.append(name)

config = LoraConfig(target_modules=target_mods)

peft_model = get_peft_model(model, config)
peft_model.print_trainable_parameters()

with open("output.txt", "w") as f:
    print(peft_model, file=f)
wrapper_model = PeftModelLightning(peft_model)

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
trainer.fit(wrapper_model, dataloader)
