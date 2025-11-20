import os
import torch
import random
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as F
import lightning as L
import wandb

from lightning.pytorch.loggers import WandbLogger
from models.MeTU import lt_MeTU
from models.SegFormer import lt_segformerb0
from datasets import CityScapes, TrainTransforms, ValTransforms, NewTrainTransforms
from torch.utils.data import DataLoader


MODEL_NAME = "New_MeTU-xxs_with_new_dec_expand_shallow_layer"

CLASS_RATIOS = {
    "0": 0.369,
    "1": 0.061,
    "2": 0.228,
    "3": 0.007,
    "4": 0.009,
    "5": 0.012,
    "6": 0.002,
    "7": 0.006,
    "8": 0.159,
    "9": 0.012,
    "10": 0.04,
    "11": 0.012,
    "12": 0.001,
    "13": 0.07,
    "14": 0.003,
    "15": 0.002,
    "16": 0.002,
    "17": 0.001,
    "18": 0.004,
}


def create_weighted_sampler(dataset, class_ratios: dict, ignore_index: int = 255):
    """
    Specific sampler for CityScapes. Total valid calss: 19, ignore_index: 255
    dataset: torch.utils.data.Dataset
    calss_ratios(dict): list of float
        pixel level ratio of each class in the dataset
    ignore_index(int): index to ignore in the loss calculation
    Each image has multiple classes, so we calculate sample weight by summing class weights.
    """

    class_weights = {}
    for class_idx, ratio in class_ratios.items():
        class_weights[int(class_idx)] = 1.0 / np.log(1.02 + ratio)
    samples_weights = []
    for idx in range(len(dataset)):
        _, mask = dataset[idx]
        mask_np = np.array(mask)
        unique_classes = np.unique(mask_np)
        sample_weight = 0.0
        for cls in unique_classes:
            if cls == ignore_index:
                continue
            sample_weight += class_weights.get(cls, 0.0)
        samples_weights.append(sample_weight)
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=samples_weights,
        num_samples=len(samples_weights),
        replacement=True,
    )
    return sampler


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
        project="CityScapes - Semantic Segmentation",
        name=MODEL_NAME,
        log_model=True,
        save_dir=f"./wandb_logs/1118/{MODEL_NAME}/",
    )

    model = lt_MeTU(
        learning_rate=5e-4,
        model_size="xxs",
        encoder_pretrained=True,
        classes=19,
    )

    # model = lt_segformerb0(learning_rate=5e-4)
    # train_tf = TrainTransforms()
    train_tf = NewTrainTransforms()
    val_tf = ValTransforms()
    ds_train = CityScapes(mode="train", transforms=train_tf)
    ds_valid = CityScapes(mode="val", transforms=val_tf)
    ds_test = CityScapes(mode="test", transforms=val_tf)
    weight_sampler = create_weighted_sampler(
        CityScapes(mode="train"), CLASS_RATIOS, ignore_index=255
    )

    train_loader = DataLoader(
        ds_train,
        batch_size=8,
        shuffle=False,  # Weigth sampler 쓰면 False로 바꿔야함.
        num_workers=4,
        worker_init_fn=seed_worker,
        sampler=weight_sampler,
    )
    val_loader = DataLoader(ds_valid, batch_size=16, shuffle=False, num_workers=4)

    trainer = L.Trainer(
        max_epochs=200,
        accelerator="auto",
        devices=1,
        precision=16,
        logger=wandb_logger,
        log_every_n_steps=50,
        callbacks=[
            # Add model checkpoint callback that also logs to wandb
            L.pytorch.callbacks.ModelCheckpoint(
                dirpath=f"checkpoints/cityscapes/1118/{MODEL_NAME}",
                filename="{epoch:02d}-{val/mIoU:.3f}",
                monitor="val/mIoU",
                mode="max",
                save_top_k=3,
            ),
            # Early stopping
            L.pytorch.callbacks.EarlyStopping(
                monitor="val/mIoU", patience=30, mode="max"
            ),
        ],
    )

    trainer.fit(model, train_loader, val_loader)

    trainer.test(model, val_loader)

    wandb.finish()
