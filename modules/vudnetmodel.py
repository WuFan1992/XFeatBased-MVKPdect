import torch
import torch.nn as nn

from modules.training.vudnet_descriptor import MobileNetDescriptorBranch, VarianceHead
from modules.training.vudnet_detector import MobileNetDetectorBranch


class VUDNetModel(nn.Module):

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
