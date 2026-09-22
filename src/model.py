import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, deeplabv3_resnet50


class NewASPP(nn.Module):
    def __init__(self, in_channels: int, atrous_rates=(6, 12, 18)):
        super().__init__()
        out_channels = 256

        convs = [
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        ]
        for rate in atrous_rates:
            convs.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels, out_channels,
                        kernel_size=3,
                        padding=rate, dilation=rate,
                        bias=False
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )

        self.convs = nn.ModuleList(convs)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        feats = [conv(x) for conv in self.convs]
        x = torch.cat(feats, dim=1)
        return self.project(x)


class NewDeepLabHead(nn.Sequential):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__(
            NewASPP(in_channels),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1),
        )


class NewDeepLabV3(nn.Module):

    def __init__(self, num_classes=4, pretrained=True):
        super().__init__()
        self.pretrained = pretrained
        weights = DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None
        self.base_model = deeplabv3_resnet50(weights=weights, weights_backbone=None, aux_loss=True)
        self.high_channels = 2048
        self.low_channels = 256
        self.num_classes = num_classes
        del self.base_model.classifier
        self.base_model.classifier = NewDeepLabHead(2048, num_classes)

    def forward(self, x):
        size = x.shape[-2:]
        features = self.base_model.backbone(x)['out']
        x = self.base_model.classifier(features)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class CombinedLoss(nn.Module):
    def __init__(self,
                 class_weights,
                 alpha: float = 1.0,
                 beta: float = 0.5,
                 eps: float = 1e-6,
                 ignore_index: int = 255):
        super().__init__()

        w = torch.as_tensor(class_weights, dtype=torch.float32)
        if w.ndim != 1 or not torch.isfinite(w).all() or (w <= 0).any():
            raise ValueError("Class weights must be finite and positive")
        self.ignore_index = ignore_index
        self.register_buffer("w_ce", w)
        self.register_buffer("w_iou", w / w.sum())
        self.alpha, self.beta, self.eps = alpha, beta, eps

        self.ce = nn.CrossEntropyLoss(weight=self.w_ce, reduction="mean", ignore_index=ignore_index)


    def _weighted_soft_iou(self, pred, target):
        valid = target != self.ignore_index
        prob = F.softmax(pred, dim=1) * valid.unsqueeze(1)
        tgt = F.one_hot(target.masked_fill(~valid, 0), pred.size(1)).permute(0, 3, 1, 2)
        tgt = tgt.float() * valid.unsqueeze(1)

        dims  = (0, 2, 3)
        inter = (prob * tgt).sum(dim=dims)
        union = prob.sum(dim=dims) + tgt.sum(dim=dims) - inter

        present = tgt.sum(dim=dims) > 0
        if not present.any():
            return torch.tensor(0.0, device=pred.device, requires_grad=False)

        iou = (inter[present] + self.eps) / (union[present] + self.eps)

        w_iou = self.w_iou[present]
        w_iou = w_iou / w_iou.sum()
        return 1.0 - (iou * w_iou).sum()


    def forward(self, pred, target):
        if not (target != self.ignore_index).any():
            return pred.sum() * 0.0
        ce_loss = self.ce(pred, target)
        iou_loss = self._weighted_soft_iou(pred, target)
        return self.alpha * ce_loss + self.beta * iou_loss
