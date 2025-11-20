import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR


def iou_component(preds, targets, num_class, ignore_idx=None):
    """
    Calcualate intersection and union value for segmentation.

    Args:
        preds (torch.Tensor): model prediction tensor(B, H, W). Each pixels in [0, num_class-1]
        targets (torch.Tensor): Ground truth mask tensor(B, H, W). Each pixels in [0, num_class-1]
        num_class (int): Number of valid classes (Except background)
        ignore_class: Ignore class when calculating IoU
    """
    if ignore_idx is not None:
        valid_mask = targets != ignore_idx
        preds = preds[valid_mask]
        targets = targets[valid_mask]

    intersection_sum = torch.zeros(num_class, dtype=torch.long, device=preds.device)
    union_sum = torch.zeros(num_class, dtype=torch.long, device=preds.device)

    preds, targets = preds.flatten(), targets.flatten()

    for class_id in range(num_class):
        is_pred = preds == class_id
        is_target = targets == class_id

        intersection = is_pred & is_target
        union = is_pred | is_target

        intersection_sum[class_id] = intersection.sum()
        union_sum[class_id] = union.sum()

    return intersection_sum, union_sum


def iou_calculation(intersection_sum, union_sum):
    """
    Calculate the stacked intersection adn union value. When union value is 0, then the IoU is 0.

    Args:
        intersection_sum(torch.tensor): The intersection value of each class.
        union_sum(torch.tensor): The union value of each class.

    """

    intersection_f = intersection_sum.float()
    union_f = union_sum.float()

    iou_per_class = torch.where(
        union_f > 0, intersection_f / union_f, torch.tensor(0.0, device=union_f.device)
    )

    valid_classes = (union_f > 0).sum().item()

    if valid_classes == 0:
        return torch.tensor(0.0, device=union_f.device)

    m_iou = iou_per_class.sum() / valid_classes

    return m_iou


def iou_calculation2(intersection_sum, union_sum):
    """
    Calculate the stacked intersection adn union value. When union value is 0, then the IoU is 0.

    Args:
        intersection_sum(torch.tensor): The intersection value of each class.
        union_sum(torch.tensor): The union value of each class.

    """

    intersection_f = intersection_sum.float()
    union_f = union_sum.float()

    iou_per_class = torch.where(
        union_f > 0, intersection_f / union_f, torch.tensor(0.0, device=union_f.device)
    )

    valid_classes = (union_f > 0).sum().item()

    if valid_classes == 0:
        return torch.tensor(0.0, device=union_f.device)

    m_iou = iou_per_class.sum() / valid_classes

    return m_iou, iou_per_class


def DiceLoss(
    logits: torch.tensor,
    targets: torch.tensor,
    num_classes: int,
    ignore_index: int = None,
    eps: float = 1e-06,
):
    """
    Retunrs DiceLoss.

    Args:
        logits (torch.tensor): [B, C, H, W] raw outputs.
        targets (torch.tensor): [B, H, W] ground truth integer masks.
        num_classes (int): Number of classes
        ignore_index (int): Ignore class index
        eps (float): epsilon value


    Returns:
        diceloss(float)
    """
    probs = F.softmax(logits, dim=1)

    valid_mask = torch.ones_like(targets, dtype=torch.bool)
    if ignore_index is not None:
        valid_mask = targets != ignore_index

    safe_targets = torch.where(valid_mask, targets, torch.zeros_like(targets))

    targets_onehot = F.one_hot(safe_targets, num_classes=num_classes)
    targets_onehot = targets_onehot.permute(0, 3, 1, 2).float()

    valid_mask = valid_mask.unsqueeze(1)  # [B, 1, H, W]
    probs = probs * valid_mask
    targets_onehot = targets_onehot * valid_mask

    dims = (0, 2, 3)
    intersection = torch.sum(probs * targets_onehot, dims)
    cardinality = torch.sum(probs + targets_onehot, dims)

    dice_score = (2.0 * intersection + eps) / (cardinality + eps)
    dice_loss = 1.0 - dice_score.mean()

    return dice_loss


def FocalLoss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    ignore_index: int = None,
    labels_smoothing: float = 0.1,
):
    ce_loss = F.cross_entropy(
        logits,
        targets,
        reduction="none",
        ignore_index=ignore_index,
        label_smoothing=labels_smoothing,
    )

    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * ce_loss

    return focal_loss.mean()


class _DiceLoss(torch.nn.Module):
    """
    Caculate DiceLoss

    Args:
        num_classes(int): Total number of classes.
        ignore_index(int): Ignored class index such as "void".
        eps(float): epsilon value.

    Returns:
        DiceLoss(float): Return diceloss in [0, 1]
    """

    def __init__(self, num_classes: int, ignore_index: int = None, eps: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, logits, targets):
        """
        Retunrs DiceLoss.

        Args:
            logits: [B, C, H, W] raw outputs.
            targets: [B, H, W] ground truth integer masks.

        Returns:
            diceloss(float)
        """
        probs = F.softmax(logits, dim=1)

        # Get valid region mask
        valid_mask = torch.ones_like(targets, dtype=torch.bool)
        if self.ignore_index is not None:
            valid_mask = targets != self.ignore_index

        # Change ignore_index as 0, but ignoring by using valid_mask
        safe_targets = torch.where(valid_mask, targets, torch.zeros_like(targets))

        # one hot encoding
        targets_onehot = F.one_hot(safe_targets, num_classes=self.num_classes)
        targets_onehot = targets_onehot.permute(0, 3, 1, 2).float()

        # Ignore_index set 0
        valid_mask = valid_mask.unsqueeze(1)  # [B, 1, H, W]
        probs = probs * valid_mask
        targets_onehot = targets_onehot * valid_mask

        # ---- (5) dice 계산 ----
        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets_onehot, dims)
        cardinality = torch.sum(probs + targets_onehot, dims)

        dice_score = (2.0 * intersection + self.eps) / (cardinality + self.eps)
        dice_loss = 1.0 - dice_score.mean()

        return dice_loss


class _FocalLoss(torch.nn.Module):
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        ignore_index: int = None,
        labels_smoothing: float = 0.1,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.labels_smoothing = labels_smoothing

    def forward(self, logits, targets):
        """
        Retunrs FocalLoss.

        Args:
            logits: [B, C, H, W] raw outputs.
            targets: [B, H, W] ground truth integer masks.

        Returns:
            focalloss(float)
        """
        ce_loss = F.cross_entropy(
            logits,
            targets,
            reduction="none",
            ignore_index=self.ignore_index,
            label_smoothing=self.labels_smoothing,
        )

        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        return focal_loss.mean()


class CosineAnnealingWithWarmupLR(CosineAnnealingLR):
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1, warmup_epochs=0):
        self.warmup_epochs = warmup_epochs
        super().__init__(optimizer, T_max, eta_min, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [
                base_lr * (self.last_epoch + 1) / self.warmup_epochs
                for base_lr in self.base_lrs
            ]
        else:
            return super().get_lr()
