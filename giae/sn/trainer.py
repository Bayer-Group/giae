import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from giae.sn.modules import Model, ClassicalModel
from torchmetrics import Metric
from giae.sn.metrics import PermutationMatrixPenalty

class AccuracyOnElement(Metric):
    def __init__(self, dist_sync_on_step=False):
        # call `self.add_state`for every internal state that is needed for the metrics computations
        # dist_reduce_fx indicates the function that should be used to reduce
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # update metric states
        assert preds.shape == target.shape

        preds = torch.argmax(preds, dim=-1)
        target = torch.argmax(target, dim=-1)
        self.correct += torch.sum(preds == target)
        self.total += target.numel()

    def compute(self):
        # compute final result
        return self.correct.float() / self.total


class AccuracyOnSet(Metric):
    def __init__(self, dist_sync_on_step=False):
        # call `self.add_state`for every internal state that is needed for the metrics computations
        # dist_reduce_fx indicates the function that should be used to reduce
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # update metric states
        assert preds.shape == target.shape
        preds = torch.argmax(preds, dim=-1)
        target = torch.argmax(target, dim=-1)
        r = torch.all(preds == target, dim=1)
        self.correct += torch.sum(r)
        self.total += r.numel()

    def compute(self):
        # compute final result
        return self.correct.float() / self.total


class Trainer(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        if hparams["use_classical"]:
            print("classical model")
            self.model = ClassicalModel(
                emb_dim=hparams["emb_dim"],
                hidden_dim=hparams["hidden_dim"],
                num_digits=hparams["num_digits"],
                num_classes=hparams["num_classes"],
            )
        else:
            print("inv model")
            self.model = Model(
                emb_dim=hparams["emb_dim"],
                hidden_dim=hparams["hidden_dim"],
                num_digits=hparams["num_digits"],
                num_classes=hparams["num_classes"],
            )
        self.perm_loss = PermutationMatrixPenalty()
        self.elmenet_acc_train = AccuracyOnElement()
        self.set_acc_train = AccuracyOnSet()
        self.elmenet_acc_val = AccuracyOnElement()
        self.set_acc_val = AccuracyOnSet()

    def forward(self, x, hard):
        y, perm, emb = self.model(x, hard)
        return y, perm, emb

    def training_step(self, batch, batch_idx):
        x, y = batch
        y, perm, _ = self(x, hard=True)
        target = torch.argmax(x, dim=-1)
        recon_loss = F.cross_entropy(input=y.view(-1, self.hparams["num_classes"]), target=target.view(-1))
        perm_loss = self.perm_loss(perm)
        loss = recon_loss + 1 * perm_loss
        self.elmenet_acc_train(y, x)
        self.set_acc_train(y, x)
        self.log("element_acc", self.elmenet_acc_train)
        self.log("set_acc", self.set_acc_train)
        self.log("perm_loss", perm_loss)
        self.log("loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y, perm, _ = self(x, hard=True)
        target = torch.argmax(x, dim=-1)
        recon_loss = F.cross_entropy(input=y.view(-1, self.hparams["num_classes"]), target=target.view(-1))
        perm_loss = self.perm_loss(perm)
        loss = recon_loss + 1 * perm_loss
        self.log("val_perm_loss", perm_loss)
        self.log("val_loss", loss)

        self.elmenet_acc_val(y, x)
        self.set_acc_val(y, x)
        self.log("element_acc_val", self.elmenet_acc_val)
        self.log("set_acc_val", self.set_acc_val)

        """y, perm, _ = self(x, hard=True)
        recon_loss = F.cross_entropy(input=y.view(-1, 10), target=target.view(-1))
        perm_loss = self.perm_loss(perm)
        loss = recon_loss + 1 * perm_loss
        self.log("val_perm_loss_hard", perm_loss)
        self.log("val_loss_hard", loss)"""

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


