import torch.nn as nn


class Segmenter(nn.Module):
    def __init__(self, backbone, seg_head):
        super(Segmenter, self).__init__()
        self.backbone = backbone
        self.seg_head = seg_head

    def forward(self, x):
        features = self.backbone(x)
        output, _ = self.seg_head(features)
        return output
