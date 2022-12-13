import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from giae.so2.modules import Model, ClassicalModel


class Trainer(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        if hparams["use_classical"]:
            print("classical model")
            self.model = ClassicalModel(hparams)
        else:
            print("inv model")
            self.model = Model(hparams)

    def forward(self, x):
        y, _, _ = self.model(x)
        return y

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = self(x)
        loss = F.mse_loss(y.squeeze(), x)
        self.log("loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = self(x)
        loss = F.mse_loss(y.squeeze(), x.squeeze())
        self.log("val_loss", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self._hparams["lr"])
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer,
            gamma=0.995,
        )
        scheduler = {
            'scheduler': lr_scheduler,
            'interval': 'epoch',
            'frequency': 2
        }
        return [optimizer], [scheduler]

