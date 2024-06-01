import torch
import torch.nn as nn

from model.modules.stage_backbone import StageBackbone
from model.modules.feature_pyramid_net import FeaturePyramidNet
from model.modules.polar_head import PolarHead


class PolarInst(nn.Module):
    def __init__(self, num_polars, num_channels, num_classes):
        super(PolarInst, self).__init__()

        self.num_classes = num_classes

        self.backbone = StageBackbone()
        self.fpn = FeaturePyramidNet(num_channels)
        self.polar_head = PolarHead(num_polars, num_channels, num_classes)

        self.distance_scales = [nn.Parameter(torch.tensor(1., dtype=torch.float)) for _ in range(5)]

    def forward(self, x):
        batch_size = x.size(0)

        backbone_outs = self.backbone(x)
        fpn_outs = self.fpn(backbone_outs['c3'], backbone_outs['c4'], backbone_outs['c5'])

        class_pred, distance_pred, centerness_pred = [], [], []
        for idx, (distance_scale, fpn_out) in enumerate(zip(self.distance_scales, fpn_outs.values())):
            head_out = self.polar_head(fpn_out)

            head_out['distance'] *= distance_scale
            head_out['distance'] = head_out['distance'].exp()

            class_pred.append(head_out['cls'].permute(0, 2, 3, 1).reshape(batch_size, -1, self.num_classes))
            distance_pred.append(head_out['distance'].permute(0, 2, 3, 1).reshape(batch_size, -1, 4))
            centerness_pred.append(head_out['centerness'].permute(0, 2, 3, 1).reshape(batch_size, -1))

        class_pred = torch.cat(class_pred, dim=1)
        distance_pred = torch.cat(distance_pred, dim=1)
        centerness_pred = torch.cat(centerness_pred, dim=1)

        return class_pred, distance_pred, centerness_pred
