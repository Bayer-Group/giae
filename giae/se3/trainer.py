import torch
import pytorch_lightning as pl

from giae.se3.modules import Model
from torch_geometric.data import Batch


from giae.sn.metrics import PermutationMatrixPenalty


class Trainer(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        self._hparams = hparams
        self.model = Model(
            hidden_dim=hparams["hidden_dim"],
            emb_dim=hparams["emb_dim"],
            num_layers=hparams["num_layers"],
            num_points=hparams["num_points"],
            encoder_nearest=hparams["encoder_nearest"],
            decoder_nearest=hparams["decoder_nearest"],
            layer_norm=not hparams["omit_layer_norm"],
            encoder_aggr="mean", decoder_aggr="mean"
        )

        self.perm_loss = PermutationMatrixPenalty()

    def forward(self, batch: Batch, tau: float = 1.0, hard: bool = False, do_rot: bool = True):
        pos_out, perm, vout, rot = self.model(data=batch, tau=tau, hard=hard, do_rot=do_rot)
        return pos_out, perm, vout, rot

    def training_step(self, batch, batch_idx):
        pos_pred, perm, _, rot = self(batch, hard=False)
        recon_loss = torch.pow(pos_pred - batch.pos, 2)
        recon_loss = torch.mean(recon_loss)
        perm_loss = self.perm_loss(perm)
        loss = recon_loss + 0.025 * perm_loss

        self.log("recon_loss", recon_loss, on_step=True)
        self.log("perm_loss", perm_loss, on_step=True)
        self.log("loss", loss, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pos_pred, perm, _, rot = self(batch, hard=False)
        recon_loss = torch.pow(pos_pred - batch.pos, 2)
        recon_loss = torch.mean(recon_loss)
        perm_loss = self.perm_loss(perm)
        loss = recon_loss + 0.025 * perm_loss

        self.log("val_recon_loss_soft", recon_loss)
        self.log("val_perm_loss_soft", perm_loss)
        self.log("val_loss", loss)

        pos_pred, perm, _, rot = self(batch, hard=False)
        recon_loss = torch.pow(pos_pred - batch.pos, 2)
        recon_loss = torch.mean(recon_loss)
        self.log("val_recon_loss_hard", recon_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self._hparams["lr"])
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer,
            gamma=0.95,
        )
        scheduler = {
            'scheduler': lr_scheduler,
            'interval': 'epoch',
            'frequency': 2
        }
        return [optimizer], [scheduler]
