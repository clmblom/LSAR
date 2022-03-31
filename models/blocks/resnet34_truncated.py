import torch.nn as nn
from torchvision import models


class ResNet34(nn.Module):
    def __init__(self, truncate_blocks=0):
        super(ResNet34, self).__init__()
        self.model = models.resnet34(pretrained=True, progress=True)
        model_list = list(self.model.children())[:-(truncate_blocks+2)]
        self.model = nn.Sequential(*model_list)

    def forward(self, x):
        return self.model(x)
