import torch
import numpy as np
import matplotlib.pyplot as plt

from models.METU import lt_MeTU
from datasets import ValTransforms
from pathlib import Path
from PIL import Image


colormap = {
    0: (0, 0, 0),  # black
    1: (128, 0, 0),  # dark red
    2: (0, 128, 0),  # dark green
    3: (0, 0, 128),  # dark blue
    4: (128, 128, 0),  # olive
    5: (128, 0, 128),  # purple
    6: (0, 128, 128),  # teal
    7: (255, 0, 0),  # red
    8: (0, 255, 0),  # green
    9: (0, 0, 255),  # blue
    10: (255, 255, 0),  # yellow
    11: (255, 0, 255),  # magenta
    12: (0, 255, 255),  # cyan
    13: (255, 128, 0),  # orange
    14: (128, 255, 0),  # lime
    15: (0, 255, 128),  # aquamarine
    16: (0, 128, 255),  # sky blue
    17: (128, 0, 255),  # violet
    18: (255, 0, 128),  # pink
    19: (128, 128, 128),  # gray
}


def decode_mask(mask_tensor):
    mask_np = mask_tensor.numpy()  # (H,W)
    h, w = mask_np.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cls in range(20):
        color_mask[mask_np == cls] = colormap[cls]
    return Image.fromarray(color_mask)


check_point_path = "/home/kdb/ws/task/CityScapes_seg/checkpoints/CityScapes/MeTU-xxs/epoch=49-val/mIoU=0.373.ckpt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = lt_MeTU.load_from_checkpoint(check_point_path)
model.eval()
model.freeze()


image_dir = Path("../../data/CityScapes/images/val")
mask_dir = Path("../../data/CityScapes/masks/val")

image = Image.open(list(image_dir.rglob("*.png"))[0]).convert("RGB")
gt_mask = Image.open(list(mask_dir.rglob("*.png"))[0])

val_transform = ValTransforms()

img, gt_mask = val_transform(image, gt_mask)

img_tensor = img.unsqueeze(0).to(device)  # (1,3,H,W)

with torch.no_grad():
    logits = model(img_tensor)
    pred_mask = logits.argmax(dim=1).squeeze(0).cpu()  # (H,W)

gt_mask = gt_mask.squeeze(0).long()
gt_color = decode_mask(gt_mask)

pred_color = decode_mask(pred_mask)


plt.subplot(1, 2, 1)
plt.imshow(pred_color)
plt.subplot(1, 2, 2)
plt.imshow(gt_color)
plt.show()
