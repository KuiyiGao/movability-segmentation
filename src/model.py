import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models import resnet50
from torchvision.models.segmentation.deeplabv3 import ASPP, DeepLabV3
from torchvision.models.segmentation.lraspp import LRASPPHead


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
        self.base_model = deeplabv3_resnet50(weights='DEFAULT', pretrained=self.pretrained)
        self.high_channels = 2048
        self.low_channels = 256
        self.num_classes = num_classes
        del self.base_model.classifier
        self.base_model.classifier = NewDeepLabHead(2048, num_classes)

    def forward(self, x):
        features = self.base_model.backbone(x)['out']  # [B,2048,H/16,W/16]
        x = self.base_model.classifier(features)
        return F.interpolate(x, scale_factor=8, mode='bilinear', align_corners=False)

'''
class CombinedLoss(nn.Module):

    def __init__(self, class_weights, alpha=1.0, beta=0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.ce_loss = nn.CrossEntropyLoss(weight=torch.tensor(class_weights))

    def _iou_loss(self, pred, target):
        pred = F.softmax(pred, dim=1)
        target_onehot = F.one_hot(target, num_classes=self.ce_loss.weight.shape[0]).permute(0, 3, 1, 2)
        intersection = (pred * target_onehot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3)) - intersection
        return 1 - (intersection / (union + 1e-6)).mean()

    def forward(self, pred, target):
        ce_loss = self.ce_loss(pred, target)
        iou_loss = self._iou_loss(pred, target)
        return self.alpha * ce_loss + self.beta * iou_loss
        '''
class CombinedLoss(nn.Module):
    def __init__(self,
                 class_weights,
                 alpha: float = 1.0,
                 beta: float = 0.5,
                 eps: float = 1e-6):
        super().__init__()

        w = torch.as_tensor(class_weights, dtype=torch.float32)
        self.register_buffer("w_ce",  w)
        self.register_buffer("w_iou", w / w.sum())
        self.alpha, self.beta, self.eps = alpha, beta, eps

        self.ce = nn.CrossEntropyLoss(weight=self.w_ce, reduction="mean")


    def _weighted_soft_iou(self, pred, target):
        prob  = F.softmax(pred, dim=1)
        tgt   = F.one_hot(target, pred.size(1)).permute(0, 3, 1, 2)
        tgt   = tgt.float()

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
        ce_loss  = self.ce(pred, target)
        iou_loss = self._weighted_soft_iou(pred, target)
        return self.alpha * ce_loss + self.beta * iou_loss

if __name__ == '__main__':
    model = NewDeepLabV3(num_classes=4)
    dummy_input = torch.randn(2, 3, 512, 512)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")

    dummy_target = torch.randint(0, 4, (2, 512, 512)).long()
    class_weights = [1.0, 2.0, 3.0, 4.0]
    criterion = CombinedLoss(class_weights)
    loss = criterion(output, dummy_target)
    print(f"Loss: {loss.item():.4f}")