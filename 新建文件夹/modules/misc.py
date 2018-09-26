import torch
import torch.nn as nn
from collections import OrderedDict


class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        """
            Global average pooling over the input's spatial dimensions
        """
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, inputs):
        in_size = inputs.size()
        return inputs.view((in_size[0], in_size[1], -1)).mean(dim=2)


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# Large Separable Convolution Block
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
class LightHeadBlock(nn.Module):
    def __init__(self, in_chs, mid_chs=64, out_chs=256, kernel_size=15):
        super(LightHeadBlock, self).__init__()
        pad = int((kernel_size - 1) / 2)

        # kernel size had better be odd number so as to avoid alignment error
        self.conv_l = nn.Sequential(OrderedDict([("conv_lu", nn.Conv2d(in_chs, mid_chs,
                                                                       kernel_size=(kernel_size, 1),
                                                                       padding=(pad, 0))),
                                                 ("conv_ld", nn.Conv2d(mid_chs, out_chs,
                                                                       kernel_size=(1, kernel_size),
                                                                       padding=(0, pad)))]))

        self.conv_r = nn.Sequential(OrderedDict([("conv_ru", nn.Conv2d(in_chs, mid_chs,
                                                                       kernel_size=(1, kernel_size),
                                                                       padding=(0, pad))),
                                                 ("conv_rd", nn.Conv2d(mid_chs, out_chs,
                                                                       kernel_size=(kernel_size, 1),
                                                                       padding=(pad, 0)))]))

    def forward(self, x):
        x_l = self.conv_l(x)
        x_r = self.conv_r(x)
        return torch.add(x_l, 1, x_r)


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# SEBlock: Squeeze & Excitation (SCSE)
#            namely, Channel-wise Attention
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
class SEBlock(nn.Module):
    def __init__(self, channel, reduct_ratio=16):
        super(SEBlock, self).__init__()
        self.channel_se = nn.Sequential(OrderedDict([("avgpool", nn.AdaptiveAvgPool2d(1)),
                                                     ("linear1", nn.Conv2d(channel, channel // reduct_ratio,
                                                                           kernel_size=1, stride=1, padding=0)),
                                                     ("relu", nn.ReLU(inplace=True)),
                                                     ("linear2", nn.Conv2d(channel // reduct_ratio, channel,
                                                                           kernel_size=1, stride=1, padding=0)),
                                                     ("score", nn.Sigmoid())]))

    def forward(self, x):
        inputs = x
        chn_se = torch.exp(self.channel_se(x))
        return torch.mul(inputs, chn_se)


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# SCSEBlock: Spatial-Channel Squeeze & Excitation (SCSE)
#            namely, Spatial-wise and Channel-wise Attention
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class SCSEBlock(nn.Module):
    def __init__(self, channel, reduct_ratio=16, is_res=False):
        super(SCSEBlock, self).__init__()
        self.is_res = is_res
        
        self.channel_se = nn.Sequential(OrderedDict([("avgpool", nn.AdaptiveAvgPool2d(1)),
                                                     ("linear1", nn.Conv2d(channel, channel // reduct_ratio,
                                                                           kernel_size=1, stride=1, padding=0)),
                                                     ("relu", nn.ReLU(inplace=True)),
                                                     ("linear2", nn.Conv2d(channel // reduct_ratio, channel,
                                                                           kernel_size=1, stride=1, padding=0)),
                                                     ("score", nn.Sigmoid())]))

        self.spatial_se = nn.Sequential(OrderedDict([("conv", nn.Conv2d(channel, 1, kernel_size=1, stride=1,
                                                                        padding=0, bias=False)),
                                                     ("score", nn.Sigmoid())]))

    def forward(self, x):
        inputs = x

        chn_se = self.channel_se(x).exp()
        spa_se = self.spatial_se(x).exp()

        if self.is_res:
            torch.mul(torch.mul(inputs, chn_se), spa_se) + inputs

        return torch.mul(torch.mul(inputs, chn_se), spa_se)


class ASPPBlock(nn.Module):
    def __init__(self, in_chs, out_chs, feat_res=(28, 28), up_ratio=2, aspp_sec=(7, 14, 21)):
        super(ASPPBlock, self).__init__()

        # --------------------------------------- #
        # 1. For image-level feature
        # --------------------------------------- #
        self.gave_pool = nn.Sequential(OrderedDict([("gavg", nn.AdaptiveAvgPool2d((1, 1))),
                                                    ("conv1_0", nn.Conv2d(in_chs, out_chs,
                                                                          kernel_size=1, stride=1, padding=0,
                                                                          groups=1, bias=False, dilation=1)),
                                                    ("up0", nn.Upsample(size=feat_res, mode='bilinear',
                                                                        align_corners=True)),
                                                    ("bn0", nn.BatchNorm2d(num_features=out_chs))]))

        # --------------------------------------- #
        # 2. Convolution: 1x1
        # --------------------------------------- #
        self.conv1x1 = nn.Sequential(OrderedDict([("conv1_1", nn.Conv2d(in_chs, out_chs, kernel_size=1,
                                                                        stride=1, padding=0, bias=False,
                                                                        groups=1, dilation=1)),
                                                  ("bn1_1", nn.BatchNorm2d(num_features=out_chs))]))

        # ------------------------------------------------- #
        # 3. Convolution: 3x3, dilation: aspp_sec[0]
        # ------------------------------------------------- #
        self.aspp_bra1 = nn.Sequential(OrderedDict([("conv2_1", nn.Conv2d(in_chs, out_chs, kernel_size=3,
                                                                          stride=1, padding=aspp_sec[0], bias=False,
                                                                          groups=1, dilation=aspp_sec[0])),
                                                    ("bn2_1", nn.BatchNorm2d(num_features=out_chs))]))

        # ------------------------------------------------- #
        # 4. Convolution: 3x3, dilation: aspp_sec[1]
        # ------------------------------------------------- #
        self.aspp_bra2 = nn.Sequential(OrderedDict([("conv2_2", nn.Conv2d(in_chs, out_chs, kernel_size=3,
                                                                          stride=1, padding=aspp_sec[1], bias=False,
                                                                          groups=1, dilation=aspp_sec[1])),
                                                    ("bn2_2", nn.BatchNorm2d(num_features=out_chs))]))

        # ------------------------------------------------- #
        # 5. Convolution: 3x3, dilation: aspp_sec[2]
        # ------------------------------------------------- #
        self.aspp_bra3 = nn.Sequential(OrderedDict([("conv2_3", nn.Conv2d(in_chs, out_chs, kernel_size=3,
                                                                          stride=1, padding=aspp_sec[2], bias=False,
                                                                          groups=1, dilation=aspp_sec[2])),
                                                    ("bn2_3", nn.BatchNorm2d(num_features=out_chs))]))

        # ------------------------------------------------- #
        # 6. down channel after concatenate
        # ------------------------------------------------- #
        self.aspp_catdown = nn.Sequential(OrderedDict([("conv_down", nn.Conv2d(5*out_chs, out_chs, kernel_size=1,
                                                                               stride=1, padding=1, bias=False,
                                                                               groups=1, dilation=1)),
                                                       ("bn_down", nn.BatchNorm2d(num_features=out_chs)),
                                                       # ("act", nn.ReLU(inplace=True)),
                                                       # ("dropout", nn.Dropout2d(p=0.2, inplace=True))
                                                       ]))

        # ------------------------------------------------- #
        # 7. up-sampling the feature-map
        # ------------------------------------------------- #
        self.upsampling = nn.Upsample(size=(int(feat_res[0]*up_ratio), int(feat_res[1]*up_ratio)),
                                      mode='bilinear', align_corners=True)

    def forward(self, x):
        out = torch.cat([self.gave_pool(x),
                         self.conv1x1(x),
                         self.aspp_bra1(x),
                         self.aspp_bra2(x),
                         self.aspp_bra3(x)], dim=1)

        return self.upsampling(self.aspp_catdown(out))


class Decoder(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, out_padding=0, bias=False):
        # TODO bias=True
        super(Decoder, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_planes, in_planes//4, 1, 1, 0, bias=bias),
                                   nn.BatchNorm2d(in_planes//4),
                                   nn.ReLU(inplace=True))

        self.tp_conv = nn.Sequential(nn.ConvTranspose2d(in_planes//4, in_planes//4, kernel_size, stride,
                                                        padding, out_padding, bias=bias),
                                     nn.BatchNorm2d(in_planes//4),
                                     nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_planes//4, out_planes, 1, 1, 0, bias=bias),
                                   nn.BatchNorm2d(out_planes),
                                   nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv1(x)
        x = self.tp_conv(x)
        x = self.conv2(x)

        return x


class CoordInfo(nn.Module):
    def __init__(self, with_r=False):
        super(CoordInfo, self).__init__()
        self.with_r = with_r

    def forward(self, x):
        """
        Add Cartesian Coordination Info to Current Tensor
        :param x: shape(N, C, H, W)
        :return:  shape(N, C+2 or C+3, H, W)
        """
        batch_size, _, height, width = x.size()

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        # 1. Meshgrid using Pytorch
        # i_coords([[[0., 0., 0., 0., 0., 0.],
        #            [1., 1., 1., 1., 1., 1.],
        #            [2., 2., 2., 2., 2., 2.],
        #            [3., 3., 3., 3., 3., 3.]]])
        #
        # j_coords([[[0., 1., 2., 3., 4., 5.],
        #            [0., 1., 2., 3., 4., 5.],
        #            [0., 1., 2., 3., 4., 5.],
        #            [0., 1., 2., 3., 4., 5.]]])
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        i_coords = torch.arange(height).repeat(1, width, 1).transpose(1, 2)  # [1, H, W]
        j_coords = torch.arange(width).repeat(1, height, 1)                  # [1, H, W]

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        # 2. Normalization (-1, 1)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        i_coords = i_coords.float() / (height - 1)
        j_coords = j_coords.float() / (width - 1)

        i_coords = i_coords * 2 - 1
        j_coords = j_coords * 2 - 1

        i_coords = i_coords.repeat(batch_size, 1, 1, 1)  # [N, 1, H, W]
        j_coords = j_coords.repeat(batch_size, 1, 1, 1)  # [N, 1, H, W]

        ret = torch.cat([x, i_coords.type_as(x), j_coords.type_as(x)], dim=1)

        if self.with_r:
            rr = torch.sqrt(torch.pow(i_coords.type_as(x) - 0.5, 2) + torch.pow(j_coords.type_as(x) - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)

        return ret


if __name__ == "__main__":
    import os
    import time

    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    net_h, net_w = 4, 6
    dummy_in = torch.randn(12, 3, net_h, net_w).requires_grad_()

    co_info = CoordInfo()

    while True:
        start_time = time.time()
        dummy_out = co_info(dummy_in)
        end_time = time.time()
        print("InceptionResNetV2 inference time: {}s".format(end_time - start_time))

