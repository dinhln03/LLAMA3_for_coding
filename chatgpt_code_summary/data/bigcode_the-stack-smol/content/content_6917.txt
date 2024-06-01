from . import sean_common as common
import torch.nn as nn
import torch
from basicsr.utils.registry import ARCH_REGISTRY

class LFF(nn.Module):
    def __init__(self, scale, n_colors, conv=common.default_conv, n_feats=64):
        super(LFF, self).__init__()

        kernel_size = 3
        n_layes = 5
        act = nn.ReLU(True)

        m_head = [conv(3, n_feats, kernel_size)]

        m_body = [
            conv(
                n_feats, n_feats, kernel_size
            ) for _ in range(n_layes)
        ]

        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            nn.Conv2d(
                n_feats, n_colors, kernel_size,
                padding=(kernel_size // 2)
            )
        ]

        self.LLF_head = nn.Sequential(*m_head)
        self.LLF_body = nn.Sequential(*m_body)
        self.LLF_tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.LLF_head(x)
        x = self.LLF_body(x)
        x = self.LLF_tail(x)
        return x


class MSRB(nn.Module):
    def __init__(self, conv=common.default_conv):
        super(MSRB, self).__init__()

        n_feats = 64
        kernel_size_1 = 3
        kernel_size_2 = 5

        self.conv_3_1 = conv(n_feats, n_feats, kernel_size_1)
        self.conv_3_2 = conv(n_feats * 2, n_feats * 2, kernel_size_1)
        self.conv_5_1 = conv(n_feats, n_feats, kernel_size_2)
        self.conv_5_2 = conv(n_feats * 2, n_feats * 2, kernel_size_2)
        self.confusion = nn.Conv2d(n_feats * 4, n_feats, 1, padding=0, stride=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        input_1 = x
        output_3_1 = self.relu(self.conv_3_1(input_1))
        output_5_1 = self.relu(self.conv_5_1(input_1))
        input_2 = torch.cat([output_3_1, output_5_1], 1)
        output_3_2 = self.relu(self.conv_3_2(input_2))
        output_5_2 = self.relu(self.conv_5_2(input_2))
        input_3 = torch.cat([output_3_2, output_5_2], 1)
        output = self.confusion(input_3)
        output += x
        return output


class Edge_Net(nn.Module):
    def __init__(self, scale, n_colors, conv=common.default_conv, n_feats=64):
        super(Edge_Net, self).__init__()

        kernel_size = 3
        act = nn.ReLU(True)
        n_blocks = 5
        self.n_blocks = n_blocks

        modules_head = [conv(3, n_feats, kernel_size)]

        modules_body = nn.ModuleList()
        for i in range(n_blocks):
            modules_body.append(
                MSRB())

        modules_tail = [
            nn.Conv2d(n_feats * (self.n_blocks + 1), n_feats, 1, padding=0, stride=1),
            conv(n_feats, n_feats, kernel_size),
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, n_colors, kernel_size)]

        self.Edge_Net_head = nn.Sequential(*modules_head)
        self.Edge_Net_body = nn.Sequential(*modules_body)
        self.Edge_Net_tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x = self.Edge_Net_head(x)
        res = x

        MSRB_out = []
        for i in range(self.n_blocks):
            x = self.Edge_Net_body[i](x)
            MSRB_out.append(x)
        MSRB_out.append(res)

        res = torch.cat(MSRB_out, 1)
        x = self.Edge_Net_tail(res)
        return x


class Net(nn.Module):
    def __init__(self, scale, res_scale, conv=common.default_conv, n_feats=64):
        super(Net, self).__init__()

        n_resblock = 40
        kernel_size = 3
        act = nn.ReLU(True)

        m_head = [conv(n_feats, n_feats, kernel_size)]

        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=res_scale
            ) for _ in range(n_resblock)
        ]

        m_tail = [conv(n_feats, 3, kernel_size)]

        self.Net_head = nn.Sequential(*m_head)
        self.Net_body = nn.Sequential(*m_body)
        self.Net_tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.Net_head(x)
        res = self.Net_body(x)
        res += x
        x = self.Net_tail(res)
        return x

@ARCH_REGISTRY.register()
class SEAN(nn.Module):
    def __init__(self,
                 n_feats,
                 scale,
                 rgb_range,
                 res_scale,
                 n_colors,
                 conv=common.default_conv):
        super(SEAN, self).__init__()

        kernel_size = 3
        act = nn.ReLU(True)

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(rgb_range, rgb_mean, rgb_std)

        # define head module
        m_LFF = [LFF(scale, n_colors, n_feats=n_feats)]

        # define body module
        m_Edge = [Edge_Net(scale, n_colors, n_feats=n_feats)]

        m_Fushion = [conv(6, n_feats, kernel_size=1)]

        # define tail module
        m_Net = [Net(scale, res_scale, n_feats=n_feats)]

        self.add_mean = common.MeanShift(rgb_range, rgb_mean, rgb_std, 1)

        self.lff = nn.Sequential(*m_LFF)
        self.edge = nn.Sequential(*m_Edge)
        self.fushion = nn.Sequential(*m_Fushion)
        self.net = nn.Sequential(*m_Net)

    def forward(self, x):
        x = self.sub_mean(x)
        low = self.lff(x)
        high = self.edge(x)
        out = torch.cat([low, high], 1)
        out = self.fushion(out)
        out = self.net(out)
        x = self.add_mean(out)
        return high, x

# import torch.nn as nn
# import torch
# from basicsr.utils.registry import ARCH_REGISTRY
#
#
# import math
#
# import torch
# import torch.nn as nn
#
#
# def default_conv(in_channels, out_channels, kernel_size, bias=True):
#     return nn.Conv2d(
#         in_channels, out_channels, kernel_size,
#         padding=(kernel_size//2), bias=bias)
#
# class MeanShift(nn.Conv2d):
#     def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
#         super(MeanShift, self).__init__(3, 3, kernel_size=1)
#         std = torch.Tensor(rgb_std)
#         self.weight.data = torch.eye(3).view(3, 3, 1, 1)
#         self.weight.data.div_(std.view(3, 1, 1, 1))
#         self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
#         self.bias.data.div_(std)
#         self.requires_grad = False
#
# class BasicBlock(nn.Sequential):
#     def __init__(
#         self, in_channels, out_channels, kernel_size, stride=1, bias=False,
#         bn=True, act=nn.ReLU(True)):
#
#         m = [nn.Conv2d(
#             in_channels, out_channels, kernel_size,
#             padding=(kernel_size//2), stride=stride, bias=bias)
#         ]
#         if bn: m.append(nn.BatchNorm2d(out_channels))
#         if act is not None: m.append(act)
#         super(BasicBlock, self).__init__(*m)
#
# class ResBlock(nn.Module):
#     def __init__(
#         self, conv, n_feat, kernel_size,
#         bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
#
#         super(ResBlock, self).__init__()
#         m = []
#         for i in range(2):
#             m.append(conv(n_feat, n_feat, kernel_size, bias=bias))
#             if bn: m.append(nn.BatchNorm2d(n_feat))
#             if i == 0: m.append(act)
#
#         self.body = nn.Sequential(*m)
#         self.res_scale = res_scale
#
#     def forward(self, x):
#         res = self.body(x).mul(self.res_scale)
#         res += x
#
#         return res
#
# class Upsampler(nn.Sequential):
#     def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):
#
#         m = []
#         if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
#             for _ in range(int(math.log(scale, 2))):
#                 m.append(conv(n_feat, 4 * n_feat, 3, bias))
#                 m.append(nn.PixelShuffle(2))
#                 if bn: m.append(nn.BatchNorm2d(n_feat))
#                 if act: m.append(act())
#         elif scale == 3:
#             m.append(conv(n_feat, 9 * n_feat, 3, bias))
#             m.append(nn.PixelShuffle(3))
#             if bn: m.append(nn.BatchNorm2d(n_feat))
#             if act: m.append(act())
#         else:
#             raise NotImplementedError
#
#         super(Upsampler, self).__init__(*m)
#
# ## add SELayer
# class SELayer(nn.Module):
#     def __init__(self, channel, reduction=16):
#         super(SELayer, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.conv_du = nn.Sequential(
#                 nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
#                 nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         y = self.avg_pool(x)
#         y = self.conv_du(y)
#         return x * y
#
# ## add SEResBlock
# class SEResBlock(nn.Module):
#     def __init__(
#         self, conv, n_feat, kernel_size, reduction,
#         bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
#
#         super(SEResBlock, self).__init__()
#         modules_body = []
#         for i in range(2):
#             modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
#             if bn: modules_body.append(nn.BatchNorm2d(n_feat))
#             if i == 0: modules_body.append(act)
#         modules_body.append(SELayer(n_feat, reduction))
#         self.body = nn.Sequential(*modules_body)
#         self.res_scale = res_scale
#
#     def forward(self, x):
#         res = self.body(x)
#         #res = self.body(x).mul(self.res_scale)
#         res += x
#
#         return res
#
#
# class LFF(nn.Module):
#     def __init__(self, scale, n_colors, conv=default_conv, n_feats=64):
#         super(LFF, self).__init__()
#
#         kernel_size = 3
#         n_layes = 5
#         act = nn.ReLU(True)
#
#         m_head = [conv(3, n_feats, kernel_size)]
#
#         m_body = [
#             conv(
#                 n_feats, n_feats, kernel_size
#             ) for _ in range(n_layes)
#         ]
#
#         m_tail = [
#             Upsampler(conv, scale, n_feats, act=False),
#             nn.Conv2d(
#                 n_feats, n_colors, kernel_size,
#                 padding=(kernel_size // 2)
#             )
#         ]
#
#         self.LLF_head = nn.Sequential(*m_head)
#         self.LLF_body = nn.Sequential(*m_body)
#         self.LLF_tail = nn.Sequential(*m_tail)
#
#     def forward(self, x):
#         x = self.LLF_head(x)
#         x = self.LLF_body(x)
#         x = self.LLF_tail(x)
#         return x
#
#
# class MSRB(nn.Module):
#     def __init__(self, conv=default_conv):
#         super(MSRB, self).__init__()
#
#         n_feats = 64
#         kernel_size_1 = 3
#         kernel_size_2 = 5
#
#         self.conv_3_1 = conv(n_feats, n_feats, kernel_size_1)
#         self.conv_3_2 = conv(n_feats * 2, n_feats * 2, kernel_size_1)
#         self.conv_5_1 = conv(n_feats, n_feats, kernel_size_2)
#         self.conv_5_2 = conv(n_feats * 2, n_feats * 2, kernel_size_2)
#         self.confusion = nn.Conv2d(n_feats * 4, n_feats, 1, padding=0, stride=1)
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         input_1 = x
#         output_3_1 = self.relu(self.conv_3_1(input_1))
#         output_5_1 = self.relu(self.conv_5_1(input_1))
#         input_2 = torch.cat([output_3_1, output_5_1], 1)
#         output_3_2 = self.relu(self.conv_3_2(input_2))
#         output_5_2 = self.relu(self.conv_5_2(input_2))
#         input_3 = torch.cat([output_3_2, output_5_2], 1)
#         output = self.confusion(input_3)
#         output += x
#         return output
#
#
# class Edge_Net(nn.Module):
#     def __init__(self, scale, n_colors, conv=default_conv, n_feats=64):
#         super(Edge_Net, self).__init__()
#
#         kernel_size = 3
#         act = nn.ReLU(True)
#         n_blocks = 5
#         self.n_blocks = n_blocks
#
#         modules_head = [conv(3, n_feats, kernel_size)]
#
#         modules_body = nn.ModuleList()
#         for i in range(n_blocks):
#             modules_body.append(
#                 MSRB())
#
#         modules_tail = [
#             nn.Conv2d(n_feats * (self.n_blocks + 1), n_feats, 1, padding=0, stride=1),
#             conv(n_feats, n_feats, kernel_size),
#             Upsampler(conv, scale, n_feats, act=False),
#             conv(n_feats, n_colors, kernel_size)]
#
#         self.Edge_Net_head = nn.Sequential(*modules_head)
#         self.Edge_Net_body = nn.Sequential(*modules_body)
#         self.Edge_Net_tail = nn.Sequential(*modules_tail)
#
#     def forward(self, x):
#         x = self.Edge_Net_head(x)
#         res = x
#
#         MSRB_out = []
#         for i in range(self.n_blocks):
#             x = self.Edge_Net_body[i](x)
#             MSRB_out.append(x)
#         MSRB_out.append(res)
#
#         res = torch.cat(MSRB_out, 1)
#         x = self.Edge_Net_tail(res)
#         return x
#
#
# class Net(nn.Module):
#     def __init__(self, res_scale, conv=default_conv, n_feats=64):
#         super(Net, self).__init__()
#
#         n_resblock = 40
#         kernel_size = 3
#         act = nn.ReLU(True)
#
#         m_head = [conv(n_feats, n_feats, kernel_size)]
#
#         m_body = [
#             ResBlock(
#                 conv, n_feats, kernel_size, act=act, res_scale=res_scale
#             ) for _ in range(n_resblock)
#         ]
#
#         m_tail = [conv(n_feats, 3, kernel_size)]
#
#         self.Net_head = nn.Sequential(*m_head)
#         self.Net_body = nn.Sequential(*m_body)
#         self.Net_tail = nn.Sequential(*m_tail)
#
#     def forward(self, x):
#         x = self.Net_head(x)
#         res = self.Net_body(x)
#         res += x
#         x = self.Net_tail(res)
#         return x
#
# @ARCH_REGISTRY.register()
# class SEAN(nn.Module):
#     def __init__(self,
#                  n_feats,
#                  scale,
#                  n_colors,
#                  rgb_range,
#                  res_scale,
#                  conv=default_conv):
#         super(SEAN, self).__init__()
#
#         rgb_mean = (0.4488, 0.4371, 0.4040)
#         rgb_std = (1.0, 1.0, 1.0)
#         self.sub_mean = MeanShift(rgb_range, rgb_mean, rgb_std)
#
#         # define head module
#         m_LFF = [LFF(scale, n_colors, n_feats=n_feats)]
#
#         # define body module
#         m_Edge = [Edge_Net(scale, n_colors, n_feats=n_feats)]
#
#         m_Fushion = [conv(6, n_feats, kernel_size=1)]
#
#         # define tail module
#         m_Net = [Net(res_scale, n_feats=n_feats)]
#
#         self.add_mean = MeanShift(rgb_range, rgb_mean, rgb_std, 1)
#
#         self.lff = nn.Sequential(*m_LFF)
#         self.edge = nn.Sequential(*m_Edge)
#         self.fushion = nn.Sequential(*m_Fushion)
#         self.net = nn.Sequential(*m_Net)
#
#     def forward(self, x):
#         x = self.sub_mean(x)
#         low = self.lff(x)
#         high = self.edge(x)
#         out = torch.cat([low, high], 1)
#         out = self.fushion(out)
#         out = self.net(out)
#         x = self.add_mean(out)
#         return high, x
