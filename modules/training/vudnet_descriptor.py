import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2


def _conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class MobileNetBackbone(nn.Module):

    def __init__(self, pretrained=True):
        super().__init__()
        backbone = mobilenet_v2(pretrained=pretrained).features

        self.stage1 = backbone[0]
        self.stage2 = nn.Sequential(backbone[1], backbone[2], backbone[3])
        self.stage3 = nn.Sequential(backbone[4], backbone[5], backbone[6])
        self.stage4 = nn.Sequential(backbone[7], backbone[8], backbone[9], backbone[10], backbone[11], backbone[12], backbone[13])
        self.stage5 = nn.Sequential(backbone[14], backbone[15], backbone[16], backbone[17])

    def forward(self, x):
        x1 = self.stage1(x)  # 1/2
        x2 = self.stage2(x1)  # 1/4
        x3 = self.stage3(x2)  # 1/8
        x4 = self.stage4(x3)  # 1/16
        x5 = self.stage5(x4)  # 1/32
        return x1, x2, x3, x4, x5


class MobileNetDescriptorBranch(nn.Module):

    def __init__(self, pretrained=True, out_channels=256):
        super().__init__()
        self.backbone = MobileNetBackbone(pretrained=pretrained)

        self.proj2 = nn.Conv2d(24, out_channels, 1)
        self.proj3 = nn.Conv2d(32, out_channels, 1)
        self.proj4 = nn.Conv2d(96, out_channels, 1)
        self.proj5 = nn.Conv2d(320, out_channels, 1)

        self.fuse = nn.Sequential(
            _conv_block(out_channels * 4, out_channels, 1, 1, 0),
            _conv_block(out_channels, out_channels, 3, 1, 1),
            nn.Conv2d(out_channels, out_channels, 1),
        )

        self.matchability_head = nn.Sequential(
            _conv_block(out_channels, out_channels, 1, 1, 0),
            _conv_block(out_channels, out_channels // 2, 1, 1, 0),
            nn.Conv2d(out_channels // 2, 1, 1),
        )

    def forward(self, x):
        _, x2, x3, x4, x5 = self.backbone(x)

        x2 = self.proj2(x2)
        x3 = F.interpolate(self.proj3(x3), size=x2.shape[-2:], mode='bilinear', align_corners=False)
        x4 = F.interpolate(self.proj4(x4), size=x2.shape[-2:], mode='bilinear', align_corners=False)
        x5 = F.interpolate(self.proj5(x5), size=x2.shape[-2:], mode='bilinear', align_corners=False)

        descriptor_feats = self.fuse(torch.cat([x2, x3, x4, x5], dim=1))
        matchability = self.matchability_head(descriptor_feats)
        return descriptor_feats, matchability


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


class VarianceHead(nn.Module):

    def __init__(self, in_channels=256):
        super().__init__()
        self.head = nn.Sequential(
            _conv_block(in_channels, in_channels, 3, 1, 1),
            _conv_block(in_channels, in_channels // 2, 3, 1, 1),
            nn.Conv2d(in_channels // 2, 1, 1),
        )

    def forward(self, x):
        return torch.sigmoid(self.head(x))


class VUDNetDescriptorModel(nn.Module):

    def __init__(self, pretrained=True, use_desc_adapter=False, train_detector=False):
        super().__init__()
        self.use_desc_adapter = use_desc_adapter
        self.train_detector = train_detector

        self.descriptor_branch = MobileNetDescriptorBranch(pretrained=pretrained, out_channels=256)
        self.detector_branch = MobileNetDetectorBranch(pretrained=pretrained)
        self.variance_head = VarianceHead(in_channels=256)

        self._set_train_stage(train_detector)

    def _set_requires_grad(self, module, requires_grad):
        for parameter in module.parameters():
            parameter.requires_grad = requires_grad

    def _set_train_stage(self, train_detector):
        self.train_detector = train_detector
        self._set_requires_grad(self.descriptor_branch, not train_detector)
        self._set_requires_grad(self.variance_head, not train_detector)
        self._set_requires_grad(self.detector_branch, train_detector)

    def set_train_detector(self, train_detector):
        self._set_train_stage(train_detector)

    def _forward_impl(self, x):
        descriptor_feats, matchability = self.descriptor_branch(x)
        variance = self.variance_head(descriptor_feats)
        detector_map = self.detector_branch(x)
        return descriptor_feats, variance, matchability, detector_map

    def forward(self, x, return_detector=False):
        feats, variance, matchability, detector_map = self._forward_impl(x)
        if return_detector:
            return feats, variance, matchability, detector_map
        return feats, variance, matchability, detector_map

    def forward_with_aux(self, x, return_detector=False):
        return self.forward(x, return_detector=return_detector)
