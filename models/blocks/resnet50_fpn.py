from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import torch.nn as nn


class ResNet50_fpn(nn.Module):
    def __init__(self):
        super(ResNet50_fpn, self).__init__()
        self.model = resnet_fpn_backbone('resnet50', pretrained=True, trainable_layers=5)

    def forward(self, x):
        return self.model(x)

