import math
import torch
import random
import torch.nn as nn

from torch.nn import init
from collections import OrderedDict


class ChannelShuffle(nn.Module):
    def __init__(self, in_chns, out_chns, kernel_size=1, groups=3):
        super(ChannelShuffle, self).__init__()
        self.groups = groups
        self.fusion = nn.Sequential(nn.Conv2d(in_channels=in_chns, out_channels=out_chns,
                                              kernel_size=kernel_size, stride=1, padding=0, bias=False),
                                    nn.BatchNorm2d(out_chns),
                                    nn.ELU(inplace=True))

    def forward(self, x):
        """
        Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]
        :param x:
        :return:
        """
        batch_size, channels, height, width = x.size()
        x.view(batch_size, self.groups, channels // self.groups, height, width).permute(0, 2, 1, 3, 4).contiguous().view(
            batch_size, channels, height, width)

        return self.fusion(x)


class PBCSABlock(nn.Module):
    def __init__(self, in_chns, reduct_ratio=16, dilation=4, is_res=True, scale=1.0):
        super(PBCSABlock, self).__init__()
        self.is_res = is_res
        self.scale = scale
        # ------------------------------------------ #
        # Channel-wise Attention Model
        # ------------------------------------------ #
        self.ch_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ch_max_pool = nn.AdaptiveMaxPool2d(1)

        self.se_block = nn.Sequential(nn.Conv2d(in_chns, in_chns // reduct_ratio,
                                                kernel_size=1, stride=1, padding=0),
                                      nn.ELU(inplace=True),
                                      nn.Conv2d(in_chns // reduct_ratio, in_chns,
                                                kernel_size=1, stride=1, padding=0))
        # self.ch_conv = nn.Sequential(nn.Conv2d(in_chns, in_chns,
        #                                        kernel_size=1, stride=1, padding=0),
        #                              nn.BatchNorm2d(in_chns))

        self.sp_conv = nn.Sequential(nn.Conv2d(in_chns, in_chns // reduct_ratio,
                                               kernel_size=1, stride=1, padding=0, bias=False),
                                     nn.Conv2d(in_chns // reduct_ratio, in_chns // reduct_ratio,
                                               kernel_size=3, stride=1, padding=dilation,
                                               dilation=dilation, bias=False),
                                     nn.Conv2d(in_chns // reduct_ratio, in_chns // reduct_ratio,
                                               kernel_size=3, stride=1, padding=dilation,
                                               dilation=dilation, bias=False),
                                     nn.Conv2d(in_chns // reduct_ratio, in_chns, kernel_size=1,
                                               stride=1, padding=0, bias=False),
                                     nn.BatchNorm2d(in_chns))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # ------------------------------------------ #
        # 1. Channel-wise Attention Model
        # ------------------------------------------ #
        res = x
        avg_p = self.se_block(self.ch_avg_pool(x))
        max_p = self.se_block(self.ch_max_pool(x))
        ch_att = avg_p + max_p

        ch_att = torch.mul(x, self.sigmoid(ch_att).exp())

        # ------------------------------------------ #
        # 2. Spatial-wise Attention Model
        # ------------------------------------------ #
        sp_att = torch.mul(x, self.sigmoid(self.sp_conv(x)).exp())

        if self.is_res:
            return sp_att + res + ch_att

        return sp_att + ch_att


class StemBlock(nn.Module):
    def __init__(self, in_chns=3, out_chns=32):
        super(StemBlock, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=in_chns, out_channels=out_chns,
                                             kernel_size=3, stride=2, padding=1, bias=False),
                                   nn.BatchNorm2d(out_chns),
                                   nn.ELU(inplace=True))

        mid_chns = int(out_chns // 4)
        self.branch1 = nn.Sequential(nn.Conv2d(in_channels=mid_chns, out_channels=mid_chns,
                                               kernel_size=1, stride=1, padding=0, bias=False),
                                     nn.BatchNorm2d(mid_chns),
                                     nn.ELU(inplace=True),

                                     nn.Conv2d(in_channels=mid_chns, out_channels=mid_chns,
                                               kernel_size=3, stride=2, padding=1, bias=False),
                                     nn.BatchNorm2d(mid_chns),
                                     nn.ELU(inplace=True)
                                     )

        self.branch2 = nn.Sequential(nn.Conv2d(in_channels=mid_chns, out_channels=mid_chns,
                                               kernel_size=1, stride=1, padding=0, bias=False),
                                     nn.BatchNorm2d(mid_chns),
                                     nn.ELU(inplace=True),

                                     nn.Conv2d(in_channels=mid_chns, out_channels=mid_chns,
                                               kernel_size=5, stride=2, padding=2, bias=False),
                                     nn.BatchNorm2d(mid_chns),
                                     nn.ELU(inplace=True)
                                     )

        self.branch3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.branch4 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        self.chn_shuffle = ChannelShuffle(in_chns=out_chns, out_chns=out_chns, kernel_size=1, groups=4)

    def forward(self, x):
        x = self.conv1(x)

        chunks = torch.chunk(x, chunks=4, dim=1)
        x0 = self.branch1(chunks[0])
        x1 = self.branch2(chunks[1])
        x2 = self.branch3(chunks[2])
        x3 = self.branch4(chunks[3])

        return self.chn_shuffle(torch.cat([x3, x1, x2, x0], dim=1))


class TransitionBlock(nn.Module):
    def __init__(self, in_chns=64, reduce_ratio=0.25, out_chns=128):
        super(TransitionBlock, self).__init__()
        mid_chns = int(in_chns // 4)

        self.branch1 = nn.Sequential(nn.Conv2d(in_channels=mid_chns, out_channels=mid_chns,
                                               kernel_size=3, stride=2, padding=1, bias=False),
                                     nn.BatchNorm2d(mid_chns),
                                     nn.ELU(inplace=True))

        self.branch2 = nn.Sequential(nn.Conv2d(in_channels=mid_chns, out_channels=mid_chns,
                                               kernel_size=(5, 1), stride=(2, 1), padding=(2, 0), bias=False),
                                     nn.BatchNorm2d(mid_chns),
                                     nn.ELU(inplace=True),

                                     nn.Conv2d(in_channels=mid_chns, out_channels=mid_chns,
                                               kernel_size=(1, 5), stride=(1, 2), padding=(0, 2), bias=False),
                                     nn.BatchNorm2d(mid_chns),
                                     nn.ELU(inplace=True))

        self.branch3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.branch4 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        self.chn_shuffle = ChannelShuffle(in_chns=in_chns, out_chns=out_chns, kernel_size=1, groups=4)

    def forward(self, x):
        chunks = torch.chunk(x, chunks=4, dim=1)
        x0 = self.branch1(chunks[0])
        x1 = self.branch2(chunks[1])
        x2 = self.branch3(chunks[2])
        x3 = self.branch4(chunks[3])

        return self.chn_shuffle(torch.cat([x3, x1, x2, x0], dim=1))


class TwoWayResBlock(nn.Module):
    def __init__(self, in_chns=32, with_relu=False):
        super(TwoWayResBlock, self).__init__()
        self.with_relu = with_relu

        mid_chns = int(in_chns // 4)

        if with_relu:
            self.relu = nn.ELU(inplace=True)

        self.branch1 = nn.Sequential(nn.Conv2d(in_channels=mid_chns, out_channels=mid_chns,
                                               kernel_size=1, stride=1, padding=0, bias=False),
                                     nn.BatchNorm2d(mid_chns),
                                     nn.ELU(inplace=True))

        self.branch2 = nn.Sequential(nn.Conv2d(in_channels=mid_chns, out_channels=mid_chns,
                                               kernel_size=3, stride=1, padding=1, bias=False),
                                     nn.BatchNorm2d(mid_chns),
                                     nn.ELU(inplace=True))

        self.branch3 = nn.Sequential(nn.Conv2d(in_channels=mid_chns, out_channels=mid_chns,
                                               kernel_size=(5, 1), stride=1, padding=(2, 0), bias=False),
                                     nn.BatchNorm2d(mid_chns),
                                     nn.ELU(inplace=True),

                                     nn.Conv2d(in_channels=mid_chns, out_channels=mid_chns,
                                               kernel_size=(1, 5), stride=1, padding=(0, 2), bias=False),
                                     nn.BatchNorm2d(mid_chns),
                                     nn.ELU(inplace=True))

        self.branch4 = nn.Sequential(nn.Conv2d(in_channels=mid_chns, out_channels=mid_chns,
                                               kernel_size=(7, 1), stride=1, padding=(3, 0), bias=False),
                                     nn.BatchNorm2d(mid_chns),
                                     nn.ELU(inplace=True),

                                     nn.Conv2d(in_channels=mid_chns, out_channels=mid_chns,
                                               kernel_size=(1, 7), stride=1, padding=(0, 3), bias=False),
                                     nn.BatchNorm2d(mid_chns),
                                     nn.ELU(inplace=True))

        self.chn_shuffle = ChannelShuffle(in_chns=in_chns, out_chns=in_chns, kernel_size=1, groups=4)

    def forward(self, x):
        residual = x
        chunks = torch.chunk(x, chunks=4, dim=1)
        x0 = self.branch1(chunks[0])
        x1 = self.branch2(chunks[1])
        x2 = self.branch3(chunks[2])
        x3 = self.branch3(chunks[3])

        if self.with_relu:
            return self.relu(self.chn_shuffle(torch.cat([x0, x2, x1, x3], dim=1)) + residual)

        return self.chn_shuffle(torch.cat([x0, x2, x1, x3], dim=1)) + residual


class ScratchNetV1(nn.Module):
    """
    Generate model architecture
    """

    def __init__(self, num_classes=2, in_size=(224, 224), width_mult=1.0):
        """
        Model initialization
        :param x_n: number of input neurons
        :type x_n: int
        """
        super(ScratchNetV1, self).__init__()

        assert in_size[0] % 8 == 0
        assert in_size[1] % 8 == 0
        self.in_size = in_size

        self.stage_repeats = [5, 9]

        # +++++++++++++++++++++++++++++++++++++++++++++++ #
        # 1. Encoders
        # +++++++++++++++++++++++++++++++++++++++++++++++ #
        self.entry = StemBlock(in_chns=3, out_chns=int(64 * width_mult))  # 1/4

        in_chns = int(64 * width_mult)
        encode_block1 = OrderedDict()
        for i in range(self.stage_repeats[0]):
            encode_block1["res_{}".format(i)] = TwoWayResBlock(in_chns=in_chns, with_relu=True)

        encode_block1["scse"] = PBCSABlock(in_chns=in_chns, reduct_ratio=16, dilation=4, is_res=True)
        # encode_block1["dropout"] = nn.Dropout2d(p=0.5)
        self.encoder1 = nn.Sequential(encode_block1)
        # -------------------------- 1/4 End -------------------------- #

        out_chns = in_chns
        in_chns = int(128 * width_mult)

        encode_block2 = OrderedDict()
        encode_block2["trans"] = TransitionBlock(in_chns=out_chns, reduce_ratio=0.25, out_chns=in_chns)
        for i in range(self.stage_repeats[1]):
            encode_block2["res_{}".format(i)] = TwoWayResBlock(in_chns=in_chns, with_relu=True)

        encode_block2["scse"] = PBCSABlock(in_chns=in_chns, reduct_ratio=16, dilation=4, is_res=True)
        # encode_block2["dropout"] = nn.Dropout2d(p=0.5)
        self.encoder2 = nn.Sequential(encode_block2)
        # -------------------------- 1/8 End -------------------------- #

        # +++++++++++++++++++++++++++++++++++++++++++++++ #
        # 2. Decoders
        # +++++++++++++++++++++++++++++++++++++++++++++++ #
        last_chns = int(width_mult*64) + int(width_mult*128)
        self.fusion = nn.Sequential(nn.Conv2d(last_chns, last_chns // 2,
                                              kernel_size=1, stride=1, padding=0, bias=False),
                                    nn.BatchNorm2d(last_chns // 2),
                                    nn.ELU(inplace=True))

        

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight.data, a=0.0, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.ConvTranspose2d):
                init.kaiming_normal_(m.weight.data, a=0.0, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight.data, a=0.0, mode='fan_in', nonlinearity='relu')
                init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        # +++++++++++++++++++++++++++++++++++++++++++++++ #
        # 1. Encoder
        # +++++++++++++++++++++++++++++++++++++++++++++++ #
        x = self.entry(x)              # [N, 64,  H/4, W/4]
        stg1 = self.encoder1(x)        # [N, 64, H/4, W/4]          <----
        stg2 = self.encoder2(stg1)     # [N, 128, H/8, W/8]         <----
        stg3 = self.encoder3(stg2)     # [N, 256, H/16, W/16]       <----
        stg3 = stg3 + self.stg1_ext(stg1)

        # +++++++++++++++++++++++++++++++++++++++++++++++ #
        # 2. Decoder
        # +++++++++++++++++++++++++++++++++++++++++++++++ #
        destg3 = stg2 + self.decoder3(stg3)    # 1/8   256
        destg2 = stg1 + self.decoder2(destg3)  # 1/4   128

        # +++++++++++++++++++++++++++++++++++++++++++++++ #
        # 3. Classifier
        # +++++++++++++++++++++++++++++++++++++++++++++++ #
        destg2 = self.tp_conv1(destg2)  # 1/2   128
        destg2 = self.conv2(destg2)
        destg2 = self.tp_conv2(destg2)  # 1/1   2
        return destg2


if __name__ == "__main__":
    import os
    import time
    import torch
    from models.scseliunet import SCSELiuNet
    from utils.loss import bootstrapped_cross_entropy2d

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    net_h, net_w = 224, 224
    dummy_in = torch.randn(1, 3, net_h, net_w).cuda().requires_grad_()
    dummy_target = torch.ones(1, net_h, net_w).cuda().long()

    model = SCSELiuNetV2(num_classes=2, in_size=(net_h, net_w), width_mult=2.0)
    model = torch.nn.DataParallel(model, device_ids=[0]).cuda()
    # model.eval()
    id_x = 0
    while True:
        id_x +=1

        start_time = time.time()
        dummy_out = model(dummy_in)
        print("Inference time: {}s".format(time.time() - start_time))