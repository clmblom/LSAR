from .blocks.resnet50_fpn import ResNet50_fpn
from .blocks.segmentation_head import SegmentationHead
from .blocks.segmenter import Segmenter


def fpn_segmenter(backbone_feature_size=256):
    backbone = ResNet50_fpn()
    seg_head = SegmentationHead(backbone_feature_size)
    segmenter = Segmenter(backbone, seg_head)
    return segmenter