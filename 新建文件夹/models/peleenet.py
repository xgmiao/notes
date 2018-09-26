import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import init
from collections import OrderedDict


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
                                      nn.ReLU(inplace=True),
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

        self.branch1 = nn.Sequential(nn.Conv2d(in_channels=out_chns, out_channels=int(out_chns//2),
                                               kernel_size=1, stride=1, padding=0, bias=False),
                                     nn.BatchNorm2d(int(out_chns//2)),
                                     nn.ELU(inplace=True),
									 
                                     nn.Conv2d(in_channels=int(out_chns//2), out_channels=out_chns,
                                               kernel_size=3, stride=2, padding=1, bias=False),
                                     nn.BatchNorm2d(out_chns),
                                     nn.ELU(inplace=True)
                                     )
        self.branch2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.branch3 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        self.fusion = nn.Sequential(nn.Conv2d(in_channels=out_chns*3, out_channels=out_chns,
                                              kernel_size=1, stride=1, padding=0, bias=False),
                                    nn.BatchNorm2d(out_chns),
                                    nn.ELU(inplace=True))

    def forward(self, x):
        x = self.conv1(x)
        x0 = self.branch1(x)
        x1 = self.branch2(x)
        x2 = self.branch3(x)
        out = torch.cat([x0, x1, x2], dim=1)
        return self.fusion(out)


class TransitionBlock(nn.Module):
    def __init__(self, chns, reduce_ratio=0.5):
        super(TransitionBlock, self).__init__()
        self.mid_chns = int(chns * reduce_ratio)

        self.branch1 = nn.Sequential(nn.Conv2d(in_channels=chns, out_channels=self.mid_chns,
                                               kernel_size=1, stride=1, padding=0, bias=False),
                                     nn.BatchNorm2d(self.mid_chns),
                                     nn.ELU(inplace=True),

                                     nn.Conv2d(in_channels=self.mid_chns, out_channels=self.mid_chns,
                                               kernel_size=3, stride=2, padding=1, bias=False),
                                     nn.BatchNorm2d(self.mid_chns),
                                     nn.ELU(inplace=True),

                                     nn.Conv2d(in_channels=self.mid_chns, out_channels=chns,
                                               kernel_size=1, stride=1, padding=0, bias=False),
                                     nn.BatchNorm2d(chns),
                                     nn.ELU(inplace=True)
                                     )
        self.branch2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.branch3 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        self.fusion = nn.Sequential(nn.Conv2d(in_channels=chns*3, out_channels=chns,
                                              kernel_size=1, stride=1, padding=0, bias=False),
                                    nn.BatchNorm2d(chns),
                                    nn.ELU(inplace=True))

    def forward(self, x):
        x0 = self.branch1(x)
        x1 = self.branch2(x)
        x2 = self.branch3(x)
        out = torch.cat([x0, x1, x2], dim=1)
        return self.fusion(out)


class TwoWayDenseBlock(nn.Module):
    def __init__(self, in_chns=32, mid_chns=16, out_chns=16, with_relu=False):
        super(TwoWayDenseBlock, self).__init__()
        self.with_relu = with_relu

        if with_relu:
            self.relu = nn.ELU(inplace=True)

        self.branch1 = nn.Sequential(nn.Conv2d(in_channels=in_chns, out_channels=mid_chns,
                                               kernel_size=1, stride=1, padding=0, bias=False),
                                     nn.BatchNorm2d(mid_chns),
                                     nn.ELU(inplace=True),

                                     nn.Conv2d(in_channels=mid_chns, out_channels=out_chns,
                                               kernel_size=3, stride=1, padding=1, bias=False),
                                     nn.BatchNorm2d(out_chns))

        self.branch2 = nn.Sequential(nn.Conv2d(in_channels=in_chns, out_channels=mid_chns,
                                               kernel_size=1, stride=1, padding=0, bias=False),
                                     nn.BatchNorm2d(mid_chns),
                                     nn.ELU(inplace=True),

                                     nn.Conv2d(in_channels=mid_chns, out_channels=out_chns,
                                               kernel_size=3, stride=1, padding=1, bias=False),
                                     nn.BatchNorm2d(out_chns),
                                     nn.ELU(inplace=True),

                                     nn.Conv2d(in_channels=out_chns, out_channels=out_chns,
                                               kernel_size=3, stride=1, padding=1, bias=False),
                                     nn.BatchNorm2d(out_chns))

        # self.scse = SCSABlock(in_chns=out_chns*2, reduct_ratio=4, is_res=True)

    def forward(self, x):
        x0 = self.branch1(x)
        x1 = self.branch2(x)
        # att = self.scse(torch.cat([x0, x1], dim=1))
        out = torch.cat([x0, x, x1], dim=1)

        if self.with_relu:
            out = self.relu(out)
        return out


class Decoder(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, out_padding=0, bias=False):
        # TODO bias=True
        super(Decoder, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_planes, in_planes//4, 1, 1, 0, bias=bias),
                                   nn.BatchNorm2d(in_planes//4),
                                   nn.ELU(inplace=True))

        self.tp_conv = nn.Sequential(nn.ConvTranspose2d(in_planes//4, in_planes//4, kernel_size, stride,
                                                        padding, out_padding, bias=bias),
                                     nn.BatchNorm2d(in_planes//4),
                                     nn.ELU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_planes//4, out_planes, 1, 1, 0, bias=bias),
                                   nn.BatchNorm2d(out_planes),
                                   nn.ELU(inplace=True))

    def forward(self, x):
        x = self.conv1(x)
        x = self.tp_conv(x)
        x = self.conv2(x)

        return x


class PeleeNet(nn.Module):
    def __init__(self, num_classes=2, in_size=(224, 224), growth_rate=32):
        super(PeleeNet, self).__init__()
        assert in_size[0] % 16 == 0
        assert in_size[1] % 16 == 0
        self.in_size = in_size

        self.last_channel = 704

        self.num_chns = [32, 0, 0, 0, 0]
        self.repeat = [3, 4, 8, 6]
        self.width_multi = [1, 2, 4, 4]

        self.half_growth_rate = int(growth_rate//2)

        # +++++++++++++++++++++++++++++++++++++++++++++++ #
        # 1. Encoders
        # +++++++++++++++++++++++++++++++++++++++++++++++ #
        self.entry = StemBlock(in_chns=3, out_chns=self.num_chns[0])

        in_chns = self.num_chns[0]

        mid_chns = int(self.half_growth_rate * self.width_multi[0] / 4) * 4
        encode_block1 = OrderedDict()
        # encode_block1["scse1"] = PBCSABlock(in_chns=in_chns, reduct_ratio=16, dilation=4, is_res=True)
        for i in range(self.repeat[0]):
            encode_block1["dens_{}".format(i)] = TwoWayDenseBlock(in_chns=in_chns, mid_chns=mid_chns,
                                                                  out_chns=self.half_growth_rate, with_relu=True)
            in_chns += 32

        encode_block1["scse2"] = PBCSABlock(in_chns=in_chns, reduct_ratio=16, dilation=4, is_res=True)
        # encode_block1["dropout"] = nn.Dropout2d(p=0.1)
        self.encoder1 = nn.Sequential(encode_block1)
        # -------------------------- 1/4 End -------------------------- #

        self.num_chns[1] = in_chns
        self.transition1 = TransitionBlock(chns=in_chns)

        mid_chns = int(self.half_growth_rate * self.width_multi[1] / 4) * 4
        encode_block2 = OrderedDict()
        for i in range(self.repeat[1]):
            encode_block2["dens_{}".format(i)] = TwoWayDenseBlock(in_chns=in_chns, mid_chns=mid_chns,
                                                                  out_chns=self.half_growth_rate, with_relu=True)
            in_chns += 32

        encode_block2["scse"] = PBCSABlock(in_chns=in_chns, reduct_ratio=16, dilation=4, is_res=True)
        # encode_block2["dropout"] = nn.Dropout2d(p=0.1)
        self.encoder2 = nn.Sequential(encode_block2)
        # -------------------------- 1/8 End -------------------------- #

        self.num_chns[2] = in_chns
        self.transition2 = TransitionBlock(chns=in_chns)

        mid_chns = int(self.half_growth_rate * self.width_multi[2] / 4) * 4
        encode_block3 = OrderedDict()

        for i in range(self.repeat[2]):
            encode_block3["dens_{}".format(i)] = TwoWayDenseBlock(in_chns=in_chns, mid_chns=mid_chns,
                                                                  out_chns=self.half_growth_rate, with_relu=True)
            in_chns += 32

        encode_block3["scse"] = PBCSABlock(in_chns=in_chns, reduct_ratio=16, dilation=4, is_res=True)
        # encode_block3["dropout"] = nn.Dropout2d(p=0.1)
        self.encoder3 = nn.Sequential(encode_block3)
        # -------------------------- 1/16 End -------------------------- #

        self.num_chns[3] = in_chns

        # +++++++++++++++++++++++++++++++++++++++++++++++ #
        # 2. Decoders
        # +++++++++++++++++++++++++++++++++++++++++++++++ #
        self.decoder3 = Decoder(self.num_chns[3], self.num_chns[2], 3, 2, 1, 1)  # 1/8
        self.decoder2 = Decoder(self.num_chns[2], self.num_chns[1], 3, 2, 1, 1)  # 1/4
        self.decoder1 = Decoder(self.num_chns[1], self.num_chns[0], 3, 1, 1, 0)

        # +++++++++++++++++++++++++++++++++++++++++++++++ #
        # 3. Classifier
        # +++++++++++++++++++++++++++++++++++++++++++++++ #
        self.tp_conv1 = nn.Sequential(nn.ConvTranspose2d(self.num_chns[0], 96, 3, 2, 1, 1),
                                      nn.BatchNorm2d(96),
                                      nn.ELU(inplace=True))  # 1/2
        self.conv2 = nn.Sequential(nn.Conv2d(96, 96, 3, 1, 1),
                                   nn.BatchNorm2d(96),
                                   nn.ELU(inplace=True),
                                   nn.Dropout2d(p=0.2))
        self.tp_conv2 = nn.ConvTranspose2d(96, num_classes, 2, 2, 0)  # 1/1

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
        x = self.entry(x)  # [N, 128,  H/4, W/4]
        stg1 = self.encoder1(x)  # [N, 320, H/4, W/4]         <----

        stg2 = self.transition1(stg1)  # [N, 192, H/8, W/8]
        stg2 = self.encoder2(stg2)  # [N, 576, H/8, W/8]         <----

        stg3 = self.transition2(stg2)  # [N, 576, H/16, W/16]
        stg3 = self.encoder3(stg3)  # [N, 768, H/16, W/16]       <----

        destg3 = stg2 + self.decoder3(stg3)  # 1/8   576
        destg2 = stg1 + self.decoder2(destg3)  # 1/4   192
        destg1 = x + self.decoder1(destg2)  # 1/4   64

        # Classifier
        out = self.tp_conv1(destg1)  # 1/2   64
        out = self.conv2(out)
        out = self.tp_conv2(out)  # 1/1   2
        return out


if __name__ == "__main__":
    import os
    import time

    os.environ["CUDA_VISIBLE_DEVICES"] = "3,2,1"

    net_h, net_w = 224, 224
    dummy_in = torch.randn(1, 3, net_h, net_w).cuda().requires_grad_()
    # dummy_target = torch.ones(1, net_h, net_w).cuda().long()

    model = PeleeNet(num_classes=1000, in_size=(net_h, net_w)).cuda()

    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.90, weight_decay=5e-4)

    while True:
        model.train()

        start_time = time.time()
        dummy_final = model(dummy_in)
        end_time = time.time()
        print("Inference time: {}s".format(end_time - start_time))

