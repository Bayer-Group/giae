import os
import logging
from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from giae.so2.trainer import Trainer
from giae.so2.data import MNISTDataModule, MNISTDataset
from giae.so2.hyperparameter import add_arguments


logging.getLogger("lightning").setLevel(logging.WARNING)


def main(hparams):
    if not os.path.exists(hparams.save_dir):
        os.makedirs(hparams.save_dir)
    if not os.path.isdir(hparams.save_dir + "/run{}/".format(hparams.id)):
        print("Creating directory")
        os.mkdir(hparams.save_dir + "/run{}/".format(hparams.id))
    print("Starting Run {}".format(hparams.id))
    checkpoint_callback = ModelCheckpoint(
        dirpath=hparams.save_dir + "/run{}/".format(hparams.id),
        save_top_k=1,
        monitor="val_loss",
        save_last=True,
    )
    lr_logger = LearningRateMonitor()
    tb_logger = TensorBoardLogger(hparams.save_dir + "/run{}/".format(hparams.id))
    model = Trainer(hparams=hparams.__dict__)
    datamodule = MNISTDataModule(
        batch_size=hparams.batch_size,
        num_workers=hparams.num_workers,
        num_eval_samples=hparams.num_eval_samples,
        file_path=hparams.file_path
    )

    trainer = pl.Trainer(
        gpus=hparams.gpus,
        max_epochs=hparams.max_epochs,
        progress_bar_refresh_rate=5 if hparams.progress_bar else 0,
        logger=tb_logger,
        checkpoint_callback=True,
        gradient_clip_val=0.5,
        callbacks=[lr_logger, checkpoint_callback],
        precision=hparams.precision,
        resume_from_checkpoint=hparams.resume_ckpt if hparams.resume_ckpt != "" else None,
        num_sanity_val_steps=0,
    )
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = add_arguments(parser)
    args = parser.parse_args()
    main(args)
