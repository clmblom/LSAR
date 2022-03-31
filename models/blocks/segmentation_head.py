import torch.nn as nn
import torch
""" From https://github.com/MhLiao/MaskTextSpotterV3/blob/master/maskrcnn_benchmark/modeling/segmentation/segmentation.py"""


class SegmentationHead(nn.Module):

    def __init__(self, ndim=256, seg_out_dim=64):

        super(SegmentationHead, self).__init__()

        self.fpn_out5 = nn.Sequential(
            conv3x3(ndim, seg_out_dim), nn.Upsample(scale_factor=8, mode="nearest")
        )
        self.fpn_out4 = nn.Sequential(
            conv3x3(ndim, seg_out_dim), nn.Upsample(scale_factor=4, mode="nearest")
        )
        self.fpn_out3 = nn.Sequential(
            conv3x3(ndim, seg_out_dim), nn.Upsample(scale_factor=2, mode="nearest")
        )
        self.fpn_out2 = conv3x3(ndim, 64)

        self.seg_out = nn.Sequential(
            conv3x3_bn_relu(4*seg_out_dim, 64, 1),
            nn.ConvTranspose2d(64, 64, 2, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 2, 2),
            nn.Sigmoid()
        )

        self.fpn_out5.apply(self.weights_init)
        self.fpn_out4.apply(self.weights_init)
        self.fpn_out3.apply(self.weights_init)
        self.fpn_out2.apply(self.weights_init)
        self.seg_out.apply(self.weights_init)

    def forward(self, x):
        p5 = self.fpn_out5(x['3'])
        p4 = self.fpn_out4(x['2'])
        p3 = self.fpn_out3(x['1'])
        p2 = self.fpn_out2(x['0'])

        cat_features = torch.cat((p5, p4, p3, p2), 1)
        out = self.seg_out(cat_features)
        return out, cat_features

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find("BatchNorm") != -1:
            m.weight.data.fill_(1.0)
            m.bias.data.fill_(1e-4)


def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=has_bias
    )


def conv3x3_bn_relu(in_planes, out_planes, stride=1, has_bias=False):
    return nn.Sequential(
        conv3x3(in_planes, out_planes, stride, has_bias),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    )