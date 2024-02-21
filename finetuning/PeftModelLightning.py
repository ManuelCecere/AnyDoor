from torch import nn
import torch
from transformers import AdamW
from pytorch_lightning import LightningModule


class PeftModelLightning(LightningModule):
    def __init__(self, peft_model, learning_rate=1e-5):
        super().__init__()
        self.peft_model = peft_model
        self.learning_rate = learning_rate

    def forward(self, **inputs):
        # Forward pass through PeftModel
        return self.peft_model(**inputs)

    def training_step(self, batch, batch_idx):
        # Forward pass and compute loss
        # we acces the wrapped model under the peft model
        wrapped_model = self.peft_model.base_model.model
        for k in wrapped_model.ucg_training:
            p = wrapped_model.ucg_training[k]["p"]
            val = wrapped_model.ucg_training[k]["val"]
            if val is None:
                val = ""
            for i in range(len(batch[k])):
                if wrapped_model.ucg_prng.choice(2, p=[1 - p, p]):
                    batch[k][i] = val

        loss, loss_dict = wrapped_model.shared_step(batch)

        wrapped_model.log_dict(
            loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True
        )

        wrapped_model.log(
            "global_step",
            self.global_step,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False,
        )

        if wrapped_model.use_scheduler:
            lr = wrapped_model.optimizers().param_groups[0]["lr"]
            wrapped_model.log(
                "lr_abs", lr, prog_bar=True, logger=True, on_step=True, on_epoch=False
            )

        return loss

    def validation_step(self, batch, batch_idx):
        # Validation step if applicable
        outputs = self.peft_model(**batch)
        val_loss = outputs.loss
        self.log("val_loss", val_loss)

    def configure_optimizers(self):
        # Define optimizer (AdamW is commonly used with Hugging Face models)
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer
