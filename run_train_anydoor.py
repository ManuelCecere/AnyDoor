import pytorch_lightning as pl
from torch.utils.data import DataLoader
from datasets.vitonhd import VitonHDDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from cldm.hack import disable_verbosity, enable_sliced_attention
from omegaconf import OmegaConf

save_memory = False
disable_verbosity()
if save_memory:
    enable_sliced_attention()

# Datasets
DConf = OmegaConf.load("./configs/datasets.yaml")
dataset = VitonHDDataset(**DConf.Train.VitonHD)


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
