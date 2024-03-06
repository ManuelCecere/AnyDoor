import argparse
import cv2
import einops
import numpy as np
import torch
import random
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from cldm.model import create_model, load_state_dict
from cldm.hack import disable_verbosity, enable_sliced_attention
from datasets.data_utils import *
from collections import namedtuple
import time
import os
from cldm.logger import ImageLogger
from torch.utils.data import DataLoader, ConcatDataset
from datasets.vitonhd import VitonHDDataset, VitonHDDataset_agnostic
from torch.utils.data import DataLoader, SubsetRandomSampler, Subset
from pytorch_lightning.callbacks import StochasticWeightAveraging
from pytorch_lightning.tuner.tuning import Tuner
from dress_code_data.dresscode_dataset import DressCodeDatasetAnyDoor
import logging

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import albumentations as A
from omegaconf import OmegaConf

logging.basicConfig(
    level=logging.INFO,
    filename="app.log",
    filemode="w",
    format="%(name)s - %(levelname)s - %(message)s",
)
seed_everything(42, workers=True)


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
learning_rate = 7.5e-8  # value found with ligthning lr_finder
sd_locked = False
only_mid_control = False
n_gpus = 1
accumulate_grad_batches = 4

# Datasets, the Viton agnostic uses a mask agnostic wrt the garment
DConf = OmegaConf.load("./configs/datasets.yaml")
viton_dataset_train = VitonHDDataset_agnostic(**DConf.Train.VitonHD)
viton_dataset_val = VitonHDDataset_agnostic(**DConf.Test.VitonHDTest)

# some of this parameters are inherited from previous scripts, and are not applied
# TODO: clean the parameters
args = argparse.Namespace(
    batch_size=batch_size,
    category="all",
    checkpoint_dir="",
    data_pairs="{}_pairs",
    dataroot="/home/ubuntu/mnt/myvolume/DressCode",
    display_count=1000,
    epochs=150,
    exp_name="",
    height=1024,
    radius=5,
    shuffle=False,
    step=100000,
    width=768,
    workers=0,
)
dresscode_dataset_train = DressCodeDatasetAnyDoor(
    args,
    dataroot_path=args.dataroot,
    phase="train",
    order="paired",
    size=(int(args.height), int(args.width)),
)

# # reduce dataset size
# total_data = len(dresscode_dataset_train)
# subset_size = total_data // 100
# print("dataset size:", subset_size)

# # Generate a random sample of indexes for the subset
# random_indexes = random.sample(range(total_data), subset_size)

# # Creating a subset of the original dataset using the randomly selected indexes
# random_subset_dataset_train = Subset(dresscode_dataset_train, random_indexes)

# TODO: make dresscode dataset val using the test folder

print("Viton Train: ", len(viton_dataset_train))
print("DressCode Train: ", len(dresscode_dataset_train))
print("Val: ", len(viton_dataset_val))


dataset_train = ConcatDataset([dresscode_dataset_train])

dataloader_train = DataLoader(
    dataset_train,
    num_workers=8,
    batch_size=batch_size,
    shuffle=False,
)
dataloader_val = DataLoader(
    viton_dataset_val,
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

with open("finetune_parameters.txt", "w") as file:
    for name, param in model.named_parameters():
        file.write(f"{name}: {param.requires_grad}\n")


# our checkpoint callback. It start at the end of every validation.
checkpoint_callback = ModelCheckpoint(
    monitor="val/loss",
    verbose=True,
    filename="epoch{epoch:01d}-step{step:03d}-val_loss{val/loss:.4f}_DressCode_noAreaCheck",
    auto_insert_metric_name=False,
)

swa_callback = StochasticWeightAveraging(swa_lrs=1e-6)
trainer = pl.Trainer(
    gpus=n_gpus,
    precision=16,
    accelerator="gpu",
    callbacks=[checkpoint_callback, swa_callback],
    progress_bar_refresh_rate=5,
    accumulate_grad_batches=accumulate_grad_batches,
    default_root_dir="./finetuning/checkpoints",
    max_epochs=3,
    val_check_interval=600,
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

# %%
# Train!
trainer.fit(model, train_dataloaders=dataloader_train, val_dataloaders=dataloader_val)
