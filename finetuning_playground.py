import cv2
import einops
import numpy as np
import torch
import random
from pytorch_lightning import seed_everything
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from cldm.hack import disable_verbosity, enable_sliced_attention
from datasets.data_utils import *
from collections import namedtuple
import time
import os


cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import albumentations as A
from omegaconf import OmegaConf
from PIL import Image


save_memory = False
disable_verbosity()
if save_memory:
    enable_sliced_attention()


config = OmegaConf.load("./configs/inference.yaml")
model_ckpt = config.pretrained_model
model_config = config.config_file

model = create_model(model_config).cpu()
model.load_state_dict(load_state_dict(model_ckpt, location="cuda"))
model = model.cuda()
ddim_sampler = DDIMSampler(model)


print([(n, type(m)) for n, m in model().named_modules()])
