import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2


class BasicLayer(nn.Module):
    """
      Conv2d -> BN -> ReLU
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        bias=False
    ):
        super().__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=bias
            ),
            nn.BatchNorm2d(out_channels, affine=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layer(x)


class VUDNetModel(nn.Module):

    def __init__(self, pretrained=True):
        super().__init__()

        self.norm = nn.InstanceNorm2d(1)

        ########################################
        # MobileNetV2 Backbone
        ########################################

        backbone = mobilenet_v2(
            pretrained=pretrained
        ).features

        # -----------------------------------
        # OS = 2
        # (N,32,H/2,W/2)
        # -----------------------------------
        self.stage1 = backbone[0]

        # -----------------------------------
        # OS = 4
        # (N,24,H/4,W/4)
        # -----------------------------------
        self.stage2 = nn.Sequential(
            backbone[1],
            backbone[2],
            backbone[3],
        )

        # -----------------------------------
        # OS = 8
        # (N,32,H/8,W/8)
        # -----------------------------------
        self.stage3 = nn.Sequential(
            backbone[4],
            backbone[5],
            backbone[6],
        )

        # -----------------------------------
        # OS = 16
        # (N,96,H/16,W/16)
        # -----------------------------------
        self.stage4 = nn.Sequential(
            backbone[7],
            backbone[8],
            backbone[9],
            backbone[10],
            backbone[11],
            backbone[12],
            backbone[13],
        )

        # -----------------------------------
        # OS = 32
        # (N,320,H/32,W/32)
        # -----------------------------------
        self.stage5 = nn.Sequential(
            backbone[14],
            backbone[15],
            backbone[16],
            backbone[17],
        )

        ########################################
        # Channel Projection
        ########################################

        self.proj3 = nn.Conv2d(32, 64, 1)
        self.proj4 = nn.Conv2d(96, 64, 1)
        self.proj5 = nn.Conv2d(320, 64, 1)

        ########################################
        # Fusion Block
        ########################################

        self.block_fusion = nn.Sequential(
            BasicLayer(64, 64, stride=1),
            BasicLayer(64, 64, stride=1),
            nn.Conv2d(64, 64, 1)
        )

        ########################################
        # Heatmap Head
        ########################################

        self.heatmap_head = nn.Sequential(
            BasicLayer(64, 64, 1, padding=0),
            BasicLayer(64, 64, 1, padding=0),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )

        ########################################
        # Keypoint Head
        ########################################

        self.keypoint_head = nn.Sequential(
            BasicLayer(64, 64, 1, padding=0),
            BasicLayer(64, 64, 1, padding=0),
            BasicLayer(64, 64, 1, padding=0),
            nn.Conv2d(64, 65, 1),
        )
        
        
        ########################################
        # Variance Head
        ########################################
        self.variance_head = nn.Sequential(
            BasicLayer(64, 64, 3, padding=1),
            BasicLayer(64, 32, 3, padding=1),
            nn.Conv2d(32, 1, 1), 
            nn.Sigmoid()
        )

    def _unfold2d(self, x, ws=8):

        B, C, H, W = x.shape

        x = x.unfold(2, ws, ws).unfold(3, ws, ws)

        x = x.reshape(
            B,
            C,
            H // ws,
            W // ws,
            ws**2
        )

        x = x.permute(0, 1, 4, 2, 3)

        return x.reshape(
            B,
            -1,
            H // ws,
            W // ws
        )

    def forward(self, x):

        ########################################
        # grayscale normalization
        ########################################

        with torch.no_grad():
            x_gray = x.mean(dim=1, keepdim=True)
            x_gray = self.norm(x_gray)

        ########################################
        # backbone
        ########################################

        x1 = self.stage1(x)
        # H/2

        x2 = self.stage2(x1)
        # H/4

        x3 = self.stage3(x2)
        # H/8
        # (N,32,H/8,W/8)

        x4 = self.stage4(x3)
        # H/16
        # (N,96,H/16,W/16)

        x5 = self.stage5(x4)
        # H/32
        # (N,320,H/32,W/32)

        ########################################
        # projection
        ########################################

        x3 = self.proj3(x3)
        x4 = self.proj4(x4)
        x5 = self.proj5(x5)

        ########################################
        # upsample to 1/8
        ########################################

        x4 = F.interpolate(
            x4,
            size=x3.shape[-2:],
            mode='bilinear',
            align_corners=False
        )

        x5 = F.interpolate(
            x5,
            size=x3.shape[-2:],
            mode='bilinear',
            align_corners=False
        )

        ########################################
        # pyramid fusion
        ########################################

        feats = self.block_fusion(
            x3 + x4 + x5
        )

        ########################################
        # heads
        ########################################

        heatmap = self.heatmap_head(feats)
        
        variance = self.variance_head(feats)

        x_gray = x.mean(dim=1, keepdim=True)
        keypoints = self.keypoint_head(self._unfold2d(x_gray, ws=8))

        return feats, keypoints, heatmap, variance
    
