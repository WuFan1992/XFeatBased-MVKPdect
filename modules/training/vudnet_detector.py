import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.training.vudnet_descriptor import MobileNetBackbone, _conv_block


class MobileNetDetectorBranch(nn.Module):

    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = MobileNetBackbone(pretrained=pretrained)

        self.proj1 = nn.Conv2d(32, 128, 1)
        self.proj2 = nn.Conv2d(24, 128, 1)

        self.fuse = nn.Sequential(
            _conv_block(256, 128, 3, 1, 1),
            _conv_block(128, 128, 3, 1, 1),
        )

        self.head = nn.Sequential(
            _conv_block(128, 64, 3, 1, 1),
            nn.Conv2d(64, 1, 1),
        )

    def forward(self, x):
        x1, x2, _, _, _ = self.backbone(x)
        x1 = self.proj1(x1)
        x2 = F.interpolate(self.proj2(x2), size=x1.shape[-2:], mode='bilinear', align_corners=False)
        fused = self.fuse(torch.cat([x1, x2], dim=1))
        fused = F.interpolate(fused, scale_factor=2, mode='bilinear', align_corners=False)
        return torch.sigmoid(self.head(fused))


class VUDNetDetectorModel(nn.Module):

    def __init__(self, pretrained=True):
        super().__init__()
        self.detector_branch = MobileNetDetectorBranch(pretrained=pretrained)

    def forward(self, x):
        return self.detector_branch(x)
