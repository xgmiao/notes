import math
import torch
import random
import torch.nn as nn

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
                                     nn.Conv2d(in_chns // reduct_ratio, 1, kernel_size=1,
                                               stride=1, padding=0, bias=False),
                                     nn.BatchNorm2d(1))
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
                                             kernel_size=7, stride=2, padding=3, bias=False),
                                   nn.BatchNorm2d(out_chns),
                                   nn.LeakyReLU(negative_slope=0.15, inplace=True))

        self.branch1 = nn.Sequential(nn.Conv2d(in_channels=out_chns, out_channels=int(out_chns//2),
                                               kernel_size=1, stride=1, padding=0, bias=False),
                                     nn.BatchNorm2d(int(out_chns//2)),
                                     nn.ELU(inplace=True),
                                     nn.Conv2d(in_channels=int(out_chns//2), out_channels=out_chns,
                                               kernel_size=5, stride=2, padding=2, bias=False),
                                     nn.BatchNorm2d(out_chns),
                                     nn.ELU(inplace=True)
                                     )
        self.branch2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.fusion = nn.Sequential(nn.Conv2d(in_channels=out_chns*2, out_channels=out_chns,
                                              kernel_size=1, stride=1, padding=0, bias=False),
                                    nn.BatchNorm2d(out_chns),
                                    nn.LeakyReLU(negative_slope=0.15, inplace=True))

    def forward(self, x):
        x = self.conv1(x)
        x0 = self.branch1(x)
        x1 = self.branch2(x)
        out = torch.cat([x0, x1], dim=1)
        return self.fusion(out)


class TransitionBlock(nn.Module):
    def __init__(self, in_chns=64, reduce_ratio=0.5, out_chns=128):
        super(TransitionBlock, self).__init__()
        self.mid_chns = int(in_chns * reduce_ratio)

        self.branch1 = nn.Sequential(nn.Conv2d(in_channels=in_chns, out_channels=self.mid_chns,
                                               kernel_size=1, stride=1, padding=0, bias=False),
                                     nn.BatchNorm2d(self.mid_chns),
                                     nn.ELU(inplace=True),

                                     nn.Conv2d(in_channels=self.mid_chns, out_channels=self.mid_chns,
                                               kernel_size=3, stride=2, padding=1, bias=False),
                                     nn.BatchNorm2d(self.mid_chns),
                                     nn.ELU(inplace=True),
                                     nn.Conv2d(in_channels=self.mid_chns, out_channels=out_chns,
                                               kernel_size=1, stride=1, padding=0, bias=False),
                                     nn.BatchNorm2d(out_chns),
                                     nn.ELU(inplace=True)
                                     )
        self.branch2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.fusion = nn.Sequential(nn.Conv2d(in_channels=int(out_chns+in_chns), out_channels=out_chns,
                                              kernel_size=1, stride=1, padding=0, bias=False),
                                    nn.BatchNorm2d(out_chns),
                                    nn.LeakyReLU(negative_slope=0.15, inplace=True))

    def forward(self, x):
        x0 = self.branch1(x)
        x1 = self.branch2(x)
        out = torch.cat([x0, x1], dim=1)
        return self.fusion(out)


class TwoWayResBlock(nn.Module):
    def __init__(self, in_chns=32, with_relu=False):
        super(TwoWayResBlock, self).__init__()
        self.with_relu = with_relu
        self.out_chns = int(in_chns//2)

        if with_relu:
            self.relu = nn.ELU(inplace=True)

        self.branch1 = nn.Sequential(nn.Conv2d(in_channels=in_chns, out_channels=self.out_chns,
                                               kernel_size=1, stride=1, padding=0, bias=False),
                                     nn.BatchNorm2d(self.out_chns),
                                     nn.ELU(inplace=True),

                                     nn.Conv2d(in_channels=self.out_chns, out_channels=self.out_chns,
                                               kernel_size=3, stride=1, padding=1, bias=False),
                                     nn.BatchNorm2d(self.out_chns),
                                     nn.LeakyReLU(negative_slope=0.15, inplace=True))

        self.branch2 = nn.Sequential(nn.Conv2d(in_channels=in_chns, out_channels=self.out_chns,
                                               kernel_size=1, stride=1, padding=0, bias=False),
                                     nn.BatchNorm2d(self.out_chns),
                                     nn.ELU(inplace=True),

                                     nn.Conv2d(in_channels=self.out_chns, out_channels=self.out_chns,
                                               kernel_size=3, stride=1, padding=1, bias=False),
                                     nn.BatchNorm2d(self.out_chns),
                                     nn.ELU(inplace=True),

                                     nn.Conv2d(in_channels=self.out_chns, out_channels=self.out_chns,
                                               kernel_size=3, stride=1, padding=1, bias=False),
                                     nn.BatchNorm2d(self.out_chns),
                                     nn.LeakyReLU(negative_slope=0.15, inplace=True))

        self.fusion = nn.Sequential(nn.Conv2d(in_channels=in_chns, out_channels=in_chns,
                                              kernel_size=1, stride=1, padding=0, bias=False),
                                    nn.BatchNorm2d(in_chns))

    def forward(self, x):
        residual = x
        x0 = self.branch1(x)
        x1 = self.branch2(x)
        out = self.fusion(torch.cat([x0, x1], dim=1))

        if self.with_relu:
            return self.relu(out + residual)

        return out + residual


class Decoder(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, out_padding=0, bias=False):
        # TODO bias=True
        super(Decoder, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_planes, in_planes//4, 1, 1, 0, bias=bias),
                                   nn.BatchNorm2d(in_planes//4),
                                   nn.LeakyReLU(negative_slope=0.15, inplace=True))

        self.tp_conv = nn.Sequential(nn.ConvTranspose2d(in_planes//4, in_planes//4, kernel_size, stride,
                                                        padding, out_padding, bias=bias),
                                     nn.BatchNorm2d(in_planes//4),
                                     nn.LeakyReLU(negative_slope=0.15, inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_planes//4, out_planes, 1, 1, 0, bias=bias),
                                   nn.BatchNorm2d(out_planes),
                                   nn.LeakyReLU(negative_slope=0.15, inplace=True))

    def forward(self, x):
        x = self.conv1(x)
        x = self.tp_conv(x)
        x = self.conv2(x)

        return x


class DecoderB(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, out_padding=0, bias=False):
        # TODO bias=True
        super(DecoderB, self).__init__()

        self.tp_conv = nn.Sequential(nn.ConvTranspose2d(in_planes, in_planes, kernel_size=kernel_size, stride=stride,
                                                        padding=padding, output_padding=out_padding,
                                                        groups=in_planes, bias=bias),
                                     nn.BatchNorm2d(in_planes),
                                     nn.ELU(inplace=True),

                                     nn.Conv2d(in_channels=in_planes, out_channels=out_planes,
                                               kernel_size=1, stride=1, padding=0, bias=False),
                                     nn.BatchNorm2d(out_planes),
                                     nn.LeakyReLU(negative_slope=0.15, inplace=True))

    def forward(self, x):
        x = self.tp_conv(x)
        return x


class SCSELiuNetV3(nn.Module):
    """
    Generate model architecture
    """

    def __init__(self, num_classes=2, in_size=(224, 224), width_mult=1.0):
        """
        Model initialization
        :param x_n: number of input neurons
        :type x_n: int
        """
        super(SCSELiuNetV3, self).__init__()

        assert in_size[0] % 16 == 0
        assert in_size[1] % 16 == 0
        self.in_size = in_size

        self.stage_repeats = [2, 3, 3]

        # +++++++++++++++++++++++++++++++++++++++++++++++ #
        # 1. Encoders
        # +++++++++++++++++++++++++++++++++++++++++++++++ #
        self.entry = StemBlock(in_chns=3, out_chns=int(64 * width_mult))  # 1/4

        in_chns = int(64 * width_mult)
        encode_block1 = OrderedDict()
        for i in range(self.stage_repeats[0]):
            encode_block1["res_{}".format(i)] = TwoWayResBlock(in_chns=in_chns, with_relu=True)

        encode_block1["scse"] = PBCSABlock(in_chns=in_chns, reduct_ratio=16, dilation=4, is_res=True)
        # encode_block1["dropout"] = nn.Dropout2d(p=0.2)
        self.encoder1 = nn.Sequential(encode_block1)
        # -------------------------- 1/4 End -------------------------- #

        out_chns = in_chns
        in_chns = int(128 * width_mult)

        encode_block2 = OrderedDict()
        encode_block2["trans"] = TransitionBlock(in_chns=out_chns, reduce_ratio=0.25, out_chns=in_chns)
        for i in range(self.stage_repeats[1]):
            encode_block2["res_{}".format(i)] = TwoWayResBlock(in_chns=in_chns, with_relu=True)

        encode_block2["scse"] = PBCSABlock(in_chns=in_chns, reduct_ratio=16, dilation=4, is_res=True)
        # encode_block2["dropout"] = nn.Dropout2d(p=0.2)
        self.encoder2 = nn.Sequential(encode_block2)
        # -------------------------- 1/8 End -------------------------- #

        out_chns = in_chns
        in_chns = int(256 * width_mult)

        encode_block3 = OrderedDict()
        encode_block3["trans"] = TransitionBlock(in_chns=out_chns, reduce_ratio=0.25, out_chns=in_chns)
        for i in range(self.stage_repeats[2]):
            encode_block3["res_{}".format(i)] = TwoWayResBlock(in_chns=in_chns, with_relu=True)

        encode_block3["scse"] = PBCSABlock(in_chns=in_chns, reduct_ratio=16, dilation=4, is_res=True)
        # encode_block3["dropout"] = nn.Dropout2d(p=0.2)
        self.encoder3 = nn.Sequential(encode_block3)
        # -------------------------- 1/16 End -------------------------- #

        # +++++++++++++++++++++++++++++++++++++++++++++++ #
        # 2. Decoders
        # +++++++++++++++++++++++++++++++++++++++++++++++ #
        self.stg1_ext = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(int(self.in_size[0] // 16),
                                                                        int(self.in_size[1] // 16))),
                                      nn.Conv2d(int(width_mult*64), int(width_mult*256),
                                                kernel_size=1, stride=1, padding=0, bias=False),
                                      nn.BatchNorm2d(int(width_mult*256)),
                                      nn.LeakyReLU(negative_slope=0.15, inplace=True))

        self.decoder1 = Decoder(int(width_mult*64), int(width_mult*64), 3, 1, 1, 0)
        self.decoder2 = Decoder(int(width_mult*128), int(width_mult*64), 3, 2, 1, 1)       # 1/4
        self.decoder3 = Decoder(int(width_mult*256), int(width_mult*128), 3, 2, 1, 1)      # 1/8

        # +++++++++++++++++++++++++++++++++++++++++++++++ #
        # 3. Classifier
        # +++++++++++++++++++++++++++++++++++++++++++++++ #
        self.tp_conv1 = nn.Sequential(nn.ConvTranspose2d(int(width_mult*64), int(width_mult*32), 3, 2, 1, 1),
                                      nn.BatchNorm2d(int(width_mult*32)),
                                      nn.LeakyReLU(negative_slope=0.15, inplace=True))  # 1/2

        self.conv2 = nn.Sequential(nn.Conv2d(int(width_mult*32), int(width_mult*32), 3, 1, 1),
                                   nn.BatchNorm2d(int(width_mult*32)),
                                   nn.ELU(inplace=True),
                                   nn.Dropout2d(p=0.3))

        self.tp_conv2 = nn.ConvTranspose2d(int(width_mult*32), num_classes, 2, 2, 0)         # 1/1

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight.data, a=0.15, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.ConvTranspose2d):
                init.kaiming_normal_(m.weight.data, a=0.15, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight.data, a=0.15, mode='fan_in', nonlinearity='leaky_relu')
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
        destg1 = x + self.decoder1(destg2)     # 1/4   64

        # +++++++++++++++++++++++++++++++++++++++++++++++ #
        # 3. Classifier
        # +++++++++++++++++++++++++++++++++++++++++++++++ #
        destg1 = self.tp_conv1(destg1)  # 1/2   64
        destg1 = self.conv2(destg1)
        destg1 = self.tp_conv2(destg1)     # 1/1   2
        return destg1


if __name__ == "__main__":
    import os
    import time
    import torch

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    net_h, net_w = 448, 448
    dummy_in = torch.randn(1, 3, net_h, net_w).cuda().requires_grad_()
    dummy_target = torch.ones(1, net_h, net_w).cuda().long()

    model = SCSELiuNetV3(num_classes=2, in_size=(net_h, net_w), width_mult=2.0)
    model = torch.nn.DataParallel(model, device_ids=[0]).cuda()
    # del model
    # torch.cuda.empty_cache()

    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.90, weight_decay=5e-4)
    # loss_fn = bootstrapped_cross_entropy2d

    while True:
        model.eval()

        start_time = time.time()
        dummy_out = model(dummy_in)
        end_time = time.time()
        print("Inference time: {}s".format(end_time - start_time))

        # optimizer.zero_grad()
        #
        # topk = 128
        # loss = loss_fn(dummy_out, dummy_target, topk)
        # # loss = torch.rand(1).cuda().requires_grad_()
        # print("> Loss: {}".format(loss.item()))
        #
        # loss.backward()
        # optimizer.step()
