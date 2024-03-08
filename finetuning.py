import cv2
import einops
import numpy as np
import torch
import random
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from cldm.model import create_model, load_state_dict
from cldm.cldm import ControlLDM
from cldm.ddim_hacked import DDIMSampler
from cldm.hack import disable_verbosity, enable_sliced_attention
from datasets.data_utils import *
import os
from torch.utils.data import DataLoader
from datasets.vitonhd import VitonHDDataset_agnostic
from torch.utils.data import DataLoader, SubsetRandomSampler
from pytorch_lightning.callbacks import StochasticWeightAveraging
from pytorch_lightning.tuner.tuning import Tuner
import logging

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import albumentations as A
from omegaconf import OmegaConf

seed_everything(42, workers=True)
logging.basicConfig(
    level=logging.WARNING,
    filename="app.log",
    filemode="w",
    format="%(name)s - %(levelname)s - %(message)s",
)


# not used, instead configure the instance of ModelCheckpoint
class CustomModelCheckpoint(pl.Callback):
    def __init__(self, save_path, save_every_n_steps, dataloader_val):
        super().__init__()
        self.save_path = save_path
        self.save_every_n_steps = save_every_n_steps
        self.steps_counter = 0
        self.dataloader_val = dataloader_val

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        self.steps_counter += 1
        if self.steps_counter % self.save_every_n_steps == 0:
            checkpoint_path = os.path.join(
                self.save_path, f"step-{self.steps_counter}.ckpt"
            )
            trainer.save_checkpoint(checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
            trainer.validate(model=pl_module, dataloaders=self.dataloader_val)

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        val_loss = metrics.get("val/loss")
        print(f"Validation Loss: {val_loss}")


save_memory = False
disable_verbosity()
if save_memory:
    enable_sliced_attention()

# Configs
resume_path = ".ckpt/epoch=1-step=8687.ckpt"
batch_size = 4
# value found with ligthinig lr_finder
learning_rate = 1e-8
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
# train_sampler = create_fractional_sampler(dataset_train, fraction=0.1)

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


# freeze all layers and unfreeze gradually to finetune
for param in model.parameters():
    param.requires_grad = False

for param in model.model.diffusion_model.out.parameters():
    param.requires_grad = True

for param in model.model.diffusion_model.output_blocks.parameters():
    param.requires_grad = True


trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad == True)
print("trainable parameters:", trainable_count)

# if you want to double check if parameters are correctly freezed
with open("finetune_parameters.txt", "w") as file:
    for name, param in model.named_parameters():
        file.write(f"{name}: {param.requires_grad}\n")


# our checkpoint callback. It start at the end of every validation.
checkpoint_callback = ModelCheckpoint(
    monitor="val/loss",
    verbose=True,
    filename="epoch{epoch:01d}-step{step:03d}-val_loss{val/loss:.4f}",
    auto_insert_metric_name=False,
)

swa_callback = StochasticWeightAveraging(swa_lrs=1e-5)

trainer = pl.Trainer(
    gpus=n_gpus,
    precision=16,
    accelerator="gpu",
    callbacks=[checkpoint_callback, swa_callback],
    progress_bar_refresh_rate=5,
    accumulate_grad_batches=accumulate_grad_batches,
    default_root_dir="./finetuning/checkpoints",
    max_epochs=3,
    val_check_interval=180,
    profiler="simple",
)


# tuner = Tuner(trainer)
# # Run learning rate finder
# lr_finder = tuner.lr_find(
#     model=model, train_dataloaders=dataloader_train, val_dataloaders=dataloader_val
# )
# # Results can be found in
# print(lr_finder.results)

# Train!
trainer.fit(model, train_dataloaders=dataloader_train, val_dataloaders=dataloader_val)
print(checkpoint_callback.best_model_score)
