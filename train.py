import os
import torch
import random
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as F
import lightning as L
import wandb

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import TQDMProgressBar
from PIL import Image
from models.METU import lt_MeTU
from datasets import CityScapes, TrainTransforms, ValTransforms
from torch.utils.data import DataLoader


MODEL_NAME = "MeTU-xxs"


class PersistentProgressBar(TQDMProgressBar):
    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        bar.leave = True  # ✅ 이전 epoch 로그 남기기
        return bar


def set_seed(seed: int = 333):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    L.seed_everything(seed, workers=True)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


if __name__ == "__main__":
    set_seed(333)
    wandb_logger = WandbLogger(
        project="CityScapes Segmentation",
        name=MODEL_NAME,
        log_model=True,
        save_dir=f"./wandb_logs/{MODEL_NAME}",
    )

    model = lt_MeTU(
        learning_rate=1e-3, model_size="xxs", encoder_pretrained=True, classes=20
    )
    train_tf = TrainTransforms()
    val_tf = ValTransforms()
    ds_train = CityScapes(mode="train", transforms=train_tf)
    ds_valid = CityScapes(mode="val", transforms=val_tf)
    ds_test = CityScapes(mode="test", transforms=val_tf)
    train_loader = DataLoader(
        ds_train, batch_size=32, shuffle=True, num_workers=4, worker_init_fn=seed_worker
    )
    val_loader = DataLoader(ds_valid, batch_size=32, shuffle=False, num_workers=4)

    trainer = L.Trainer(
        max_epochs=50,
        accelerator="auto",
        devices=1,
        precision=16,
        logger=wandb_logger,
        log_every_n_steps=50,
        callbacks=[
            PersistentProgressBar(),
            # Add model checkpoint callback that also logs to wandb
            L.pytorch.callbacks.ModelCheckpoint(
                dirpath=f"checkpoints/CityScapes/{MODEL_NAME}",
                filename="{epoch:02d}-{val/mIoU:.3f}",
                monitor="val/mIoU",
                mode="max",
                save_top_k=3,
            ),
            # Early stopping
            L.pytorch.callbacks.EarlyStopping(
                monitor="val/loss", patience=10, mode="min"
            ),
        ],
    )

    trainer.fit(model, train_loader, val_loader)

    trainer.test(model, val_loader)

    wandb.finish()
