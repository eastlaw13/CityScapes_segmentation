import os
import sys
import timm
import torch
import warnings
import lightning as L
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from transformers import logging

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from util import (
    iou_component,
    iou_calculation,
    DiceLoss,
    FocalLoss,
    CosineAnnealingWithWarmupLR,
)

warnings.filterwarnings("ignore")
logging.set_verbosity_warning()
logging.set_verbosity_error()


class MobileViTEncoder(nn.Module):
    """
    Pretrained MobileViT v1 with Imagnet-1k based Encoder from timm API.

    Args:
        model_size (str): 'xxs', 'xs', 's'. Current verison only can use xxs ver.
        pretrained (bool): If True, load ImageNet pretrained weights
    Returns:
        model(nn.Module)
    """

    def __init__(self, model_size="xxs", pretrained=True):
        super().__init__()
        self.model_size = model_size
        self.pretrained = pretrained
        self.full_model = timm.create_model(
            f"mobilevit_{self.model_size}", pretrained=self.pretrained
        )
        self.stem = self.full_model.stem
        self.stage0 = self.full_model.stages[0]
        self.stage1 = self.full_model.stages[1]
        self.stage2 = self.full_model.stages[2]
        self.stage3 = self.full_model.stages[3]
        self.stage4 = self.full_model.stages[4]
        self.final_conv = self.full_model.final_conv

    def forward(self, x):

        skips_features = []
        out = self.stem(x)
        skips_features.append(out)
        out = self.stage0(out)
        skips_features.append(out)

        out = self.stage1(out)
        skips_features.append(out)

        out = self.stage2(out)
        skips_features.append(out)

        out = self.stage3(out)
        skips_features.append(out)

        out = self.stage4(out)
        skips_features.append(out)

        out = self.final_conv(out)

        return out, skips_features


class DWConv(nn.Module):
    """
    Depthwise separable Convoltuion block.

    Args:
        in_ch (int): Input channel size.
        out_ch (int): Output channel size.
        kernel (int): Kernel size. Default value is 3.
        padding (int): Padding size. Default value is 1.
        dilation (int): dilation value. Default value is 1.

    Returns:
        Depthwise convolution block (nn.Module)
    """

    def __init__(self, in_ch, out_ch, kernel=3, padding=1, dilation=1):
        super().__init__()
        effective_padding = padding * dilation
        self.depth = nn.Conv2d(
            in_ch,
            in_ch,
            kernel,
            padding=effective_padding,
            groups=in_ch,
            bias=False,
            dilation=dilation,
        )
        self.point = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.depth(x)
        x = self.point(x)
        x = self.bn(x)
        return x


class SEBlock(nn.Module):
    def __init__(self, ch, r=4):
        super().__init__()
        self.fc1 = nn.Conv2d(ch, max(ch // r, 1), 1)
        self.fc2 = nn.Conv2d(max(ch // r, 1), ch, 1)

    def forward(self, x):
        s = x.mean((2, 3), keepdim=True)
        s = F.relu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s))
        return x * s


class ImprovedDecoderBlock(nn.Module):
    """
    Improved decoder block that respects in_ch, skip_ch, out_ch.
    - Upsamples decoder feature x to match skip spatial size (if skip provided)
    - Projects skip to out_ch via 1x1
    - Concatenates [x, skip_proj] -> channel reduction via 1x1
    - Refinement via DWConv x2 + SE + residual (residual built from in_ch -> out_ch)
    """

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        if skip_ch is None:
            self.upsample = nn.ConvTranspose2d(
                in_ch, in_ch, kernel_size=2, stride=2, bias=False
            )
        else:
            self.up_conv = nn.ConvTranspose2d(
                in_ch, in_ch, kernel_size=2, stride=2, bias=False
            )
            self.up_bn = nn.BatchNorm2d(in_ch)

        self.skip_proj = (
            nn.Conv2d(skip_ch, out_ch, kernel_size=1) if skip_ch is not None else None
        )

        self.reduce_decoder = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)

        self.fuse_conv = nn.Conv2d(
            out_ch * (2 if skip_ch is not None else 1),
            out_ch,
            kernel_size=1,
            bias=False,
        )
        self.bn_fuse = nn.BatchNorm2d(out_ch)

        self.dw1 = DWConv(out_ch, out_ch, dilation=2, padding=1)
        self.act1 = nn.ReLU(inplace=True)
        self.dw2 = DWConv(out_ch, out_ch, dilation=4, padding=1)
        self.act2 = nn.ReLU(inplace=True)

        self.se = SEBlock(out_ch, r=4)

        self.res_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn_res = nn.BatchNorm2d(out_ch)

    def forward(self, x, skip=None):

        if skip is not None and self.skip_proj is not None:

            x_up = F.relu(self.up_bn(self.up_conv(x)), inplace=True)
            x_up = F.interpolate(
                x_up, size=skip.shape[2:], mode="bilinear", align_corners=False
            )

            x_red = self.reduce_decoder(x_up)  # (B, out_ch, hs, ws)

            skip_p = self.skip_proj(skip)  # (B, out_ch, hs, ws)

            fused = torch.cat([x_red, skip_p], dim=1)  # (B, 2*out_ch, hs, ws)
            fused = self.bn_fuse(self.fuse_conv(fused))  # (B, out_ch, hs, ws)

            res_input = x_up
        else:
            x_up = F.relu(self.bn_fuse(self.upsample(x)), inplace=True)

            fused = self.reduce_decoder(x_up)
            fused = self.bn_fuse(self.fuse_conv(fused))

            res_input = x_up

        y = self.dw1(fused)
        y = self.act1(y)
        y = self.dw2(y)
        y = self.act2(y)

        y = self.se(y)

        res = self.res_conv(res_input)
        res = self.bn_res(res)

        out = F.relu(y + res, inplace=True)
        return out


class MeTU(nn.Module):
    def __init__(self, model_size="xxs", encoder_pretrained=True, classes=1):
        super().__init__()
        self.encoder = MobileViTEncoder(model_size, encoder_pretrained)
        self.classes = classes

        # without final_conv, channels sizes: [stem, stage0, stage1, stage2, stage3, stage4]
        if model_size == "xxs":
            skip_channels = [16, 16, 24, 48, 64, 80]
        elif model_size == "xs":
            skip_channels = [16, 32, 48, 64, 80, 96]
        elif model_size == "s":
            skip_channels = [16, 32, 64, 96, 128, 160]
        else:
            e = print(
                "[WARNING] No matched size pretrained MobileViT model. Check the model_size input."
            )
            return e

        bottleneck_ch = self.encoder.final_conv.out_channels

        decoder_config = []
        prev_ch = bottleneck_ch
        for skip_ch in reversed(skip_channels):
            out_ch = max(skip_ch, prev_ch // 2)
            decoder_config.append((prev_ch, skip_ch, out_ch))
            prev_ch = out_ch

        self.decoder_blocks = nn.ModuleList(
            [
                ImprovedDecoderBlock(
                    in_ch=in_ch,
                    skip_ch=skip_ch,
                    out_ch=out_ch,
                )
                for in_ch, skip_ch, out_ch in decoder_config
            ]
        )

        self.out_conv = nn.Conv2d(decoder_config[-1][2], self.classes, kernel_size=1)

        for param in self.encoder.parameters():
            param.requires_grad = True

        # for name, param in self.encoder.named_parameters():
        #     if any(
        #         x in name
        #         for x in [
        #             "stem",
        #             "stages.0",
        #             "stages.1",
        #             "stages.2",
        #             "stages.3",
        #             "stages.4",
        #             "final_conv",
        #         ]
        #     ):
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = False

        # For update batch norms in shallow layer.
        self.encoder.stem.train()
        self.encoder.stage0.train()
        self.encoder.stage1.train()
        self.encoder.stage2.train()
        self.encoder.stage3.train()
        self.encoder.stage4.train()
        self.encoder.final_conv.train()

        # For freeze batch norms in deep layer.
        # self.encoder.stage4.eval()
        # self.encoder.final_conv.eval()

    def forward(self, x):
        input_h, input_w = x.shape[2], x.shape[3]
        bottleneck, skips = self.encoder(x)
        x = bottleneck
        for i, dec_block in enumerate(self.decoder_blocks):
            skip_feat = skips[-(i + 1)]
            x = dec_block(x, skip_feat)

        x = F.interpolate(
            x, size=(input_h, input_w), mode="bilinear", align_corners=False
        )
        out = self.out_conv(x)
        return out


class lt_MeTU(L.LightningModule):
    def __init__(
        self,
        learning_rate,
        model_size="xxs",
        encoder_pretrained=True,
        classes=1,
    ):
        super().__init__()
        self.model_size = model_size
        self.encoder_pretrained = encoder_pretrained
        self.classes = classes
        self.learning_rate = learning_rate
        self.save_hyperparameters()
        self._init_iou_components()
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.ignore_index = 255
        self.model = MeTU(
            model_size=self.model_size,
            encoder_pretrained=self.encoder_pretrained,
            classes=self.classes,
        )

    def _init_iou_components(self):
        num_classes = self.classes

        self.register_buffer(
            "train_intersections", torch.zeros(num_classes, dtype=torch.long)
        )
        self.register_buffer("train_unions", torch.zeros(num_classes, dtype=torch.long))

        self.register_buffer(
            "val_intersections", torch.zeros(num_classes, dtype=torch.long)
        )
        self.register_buffer("val_unions", torch.zeros(num_classes, dtype=torch.long))

        self.register_buffer(
            "test_intersections", torch.zeros(num_classes, dtype=torch.long)
        )
        self.register_buffer("test_unions", torch.zeros(num_classes, dtype=torch.long))

    def _reset_iou_components(self, stage):
        if stage == "train":
            self.train_intersections.zero_()
            self.train_unions.zero_()
        elif stage == "val":
            self.val_intersections.zero_()
            self.val_unions.zero_()
        elif stage == "test":
            self.test_intersections.zero_()
            self.test_unions.zero_()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        imgs, masks = batch
        logits = self(imgs)
        ce_loss = F.cross_entropy(
            logits,
            masks,
            ignore_index=self.ignore_index,
        )

        dice_loss = DiceLoss(logits, masks, 19, self.ignore_index)
        # focal_loss = FocalLoss(
        #     logits=logits,
        #     targets=msks,
        #     alpha=0.25,
        #     gamma=2.0,
        #     ignore_index=self.ignore_index,
        #     labels_smoothing=0.1,
        # )

        loss = 0.5 * ce_loss + 0.5 * dice_loss
        preds = logits.argmax(dim=1)
        inter, union = iou_component(
            preds, masks, self.classes, ignore_idx=self.ignore_index
        )

        self.train_unions += union
        self.train_intersections += inter

        self.log("train/loss", loss, on_step=False, on_epoch=True)
        self.log("train/lr", self.optimizers().param_groups[0]["lr"])

        return loss

    def on_train_epoch_end(self):
        m_iou = iou_calculation(self.train_intersections, self.train_unions)
        self.log("train/mIoU", m_iou, on_epoch=True)
        self._reset_iou_components("train")

    def validation_step(self, batch, batch_idx):
        imgs, masks = batch
        logits = self(imgs)
        ce_loss = F.cross_entropy(
            logits,
            masks,
            ignore_index=self.ignore_index,
        )
        dice_loss = DiceLoss(logits, masks, 19, self.ignore_index)
        # focal_loss = FocalLoss(
        #     logits=logits,
        #     targets=msks,
        #     alpha=0.25,
        #     gamma=2.0,
        #     ignore_index=self.ignore_index,
        #     labels_smoothing=0.1,
        # )

        loss = 0.5 * ce_loss + 0.5 * dice_loss

        preds = torch.argmax(logits, dim=1)
        inter, union = iou_component(
            preds, masks, self.classes, ignore_idx=self.ignore_index
        )
        self.val_intersections += inter.to(self.device)
        self.val_unions += union.to(self.device)

        self.validation_step_outputs.append(
            {
                "loss": loss.detach(),
            }
        )

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            return

        avg_loss = torch.stack([x["loss"] for x in self.validation_step_outputs]).mean()
        self.log("val/loss", avg_loss, on_epoch=True)

        m_iou = iou_calculation(self.val_intersections, self.val_unions)

        self.log("val/mIoU", m_iou, on_epoch=True)
        self._reset_iou_components("val")

    def test_step(self, batch, batch_idx):
        imgs, masks = batch
        logits = self(imgs)

        ce_loss = F.cross_entropy(
            logits,
            masks,
            ignore_index=self.ignore_index,
        )
        dice_loss = DiceLoss(logits, masks, 19, self.ignore_index)
        # focal_loss = FocalLoss(
        #     logits=logits,
        #     targets=msks,
        #     alpha=0.25,
        #     gamma=2.0,
        #     ignore_index=self.ignore_index,
        #     labels_smoothing=0.1,
        # )
        # )
        loss = 0.5 * ce_loss + 0.5 * dice_loss

        preds = logits.argmax(dim=1)

        self.test_step_outputs.append({"loss": loss.detach()})
        inter, union = iou_component(
            preds, masks, self.classes, ignore_idx=self.ignore_index
        )

        self.test_intersections += inter
        self.test_unions += union
        return loss

    def on_test_epoch_end(self):
        if not self.test_step_outputs:
            return
        avg_loss = torch.stack([x["loss"] for x in self.test_step_outputs]).mean()

        self.log("test/loss", avg_loss, on_epoch=True)
        self._reset_iou_components("test")

    def configure_optimizers(self):
        encoder_params = []
        decoder_params = []

        for name, param in self.model.named_parameters():
            # Append encoder parameters for updating with lower learning rate
            if any(
                x in name
                for x in [
                    "encoder.stem",
                    "encoder.stage0",
                    "encoder.stage1",
                    "encoder.stage2",
                    "encoder.stage3",
                    "encoder.stage4",
                    "encoder.final_conv",
                ]
            ):
                encoder_params.append((name, param))
            # Append encoder parameters that are frozen
            # elif any(
            #     x in name
            #     for x in [
            #         "encoder.stage4",
            #         "encoder.final_conv",
            #     ]
            # ):
            # elif:
            #     continue

            else:
                decoder_params.append((name, param))

        def split_wd(param_list, base_lr):
            wd, nwd = [], []
            for name, p in param_list:
                if p.ndim <= 1 or "bias" in name or "norm" in name.lower():
                    nwd.append(p)
                else:
                    wd.append(p)

            return [
                {"params": wd, "weight_decay": 1e-4, "lr": base_lr},
                {"params": nwd, "weight_decay": 0.0, "lr": base_lr},
            ]

        shallow_lr = 1e-04
        optim_gropups = []
        optim_gropups += split_wd(encoder_params, shallow_lr)
        optim_gropups += split_wd(decoder_params, self.learning_rate)

        optimizer = torch.optim.AdamW(optim_gropups, betas=(0.9, 0.999))
        scheduler = CosineAnnealingWithWarmupLR(
            optimizer, T_max=200, eta_min=1e-6, warmup_epochs=10
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val/loss"},
        }


if __name__ == "__main__":
    from torchinfo import summary

    model = lt_MeTU(
        model_size="xxs", encoder_pretrained=True, learning_rate=1e-3, classes=19
    )
    summary(model, (1, 3, 512, 1024))
