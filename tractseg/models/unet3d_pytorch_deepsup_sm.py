
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

from tractseg.libs.pytorch_utils import conv3d
from tractseg.libs.pytorch_utils import deconv3d


class UNet3D_Pytorch_DeepSup_sm(torch.nn.Module):
    def __init__(self, n_input_channels=3, n_classes=7, n_filt=64, batchnorm=False,
                 dropout=False, upsample="trilinear"):
        super(UNet3D_Pytorch_DeepSup_sm, self).__init__()
        self.in_channel = n_input_channels
        self.n_classes = n_classes
        self.dropout = dropout

        # Down 1
        self.contr_1_1 = conv3d(n_input_channels, n_filt)
        self.contr_1_2 = conv3d(n_filt, n_filt)
        self.pool_1 = nn.MaxPool3d((2, 2, 2))

        # Down 2
        self.contr_2_1 = conv3d(n_filt, n_filt * 2)
        self.contr_2_2 = conv3d(n_filt * 2, n_filt * 2)
        self.pool_2 = nn.MaxPool3d((2, 2, 2))

        # Down 3
        self.contr_3_1 = conv3d(n_filt * 2, n_filt * 4)
        self.contr_3_2 = conv3d(n_filt * 4, n_filt * 4)
        self.pool_3 = nn.MaxPool3d((2, 2, 2))

        # Bottleneck
        self.dropout = nn.Dropout(p=0.4)

        self.encode_1 = conv3d(n_filt * 4, n_filt * 4)
        self.encode_2 = conv3d(n_filt * 4, n_filt * 8)
        self.deconv_1 = deconv3d(n_filt * 8, n_filt * 4, kernel_size=2, stride=2)

        # Up 1
        self.expand_2_1 = conv3d(n_filt * 4 + n_filt * 4, n_filt * 4, stride=1)
        self.expand_2_2 = conv3d(n_filt * 4, n_filt * 4, stride=1)
        self.deconv_3 = deconv3d(n_filt * 4, n_filt * 2, kernel_size=2, stride=2)

        self.output_2 = nn.Conv3d(n_filt * 4 + n_filt * 4, n_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.output_2_up = nn.Upsample(scale_factor=2, mode=upsample)  # does only upscale width and height

        # Up 2
        self.expand_3_1 = conv3d(n_filt * 2 + n_filt * 2, n_filt * 2, stride=1)
        self.expand_3_2 = conv3d(n_filt * 2, n_filt * 2, stride=1)
        self.deconv_4 = deconv3d(n_filt * 2, n_filt * 1, kernel_size=2, stride=2)

        self.output_3 = nn.Conv3d(n_filt * 2 + n_filt * 2, n_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.output_3_up = nn.Upsample(scale_factor=2, mode=upsample)  # does only upscale width and height

        # Up 3
        self.expand_4_1 = conv3d(n_filt + n_filt * 1, n_filt, stride=1)
        self.expand_4_2 = conv3d(n_filt, n_filt, stride=1)

        # no activation function, because is in LossFunction (...WithLogits)
        self.conv_5 = nn.Conv3d(n_filt, n_classes, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, inpt):
        contr_1_1 = self.contr_1_1(inpt)
        contr_1_2 = self.contr_1_2(contr_1_1)
        pool_1 = self.pool_1(contr_1_2)

        contr_2_1 = self.contr_2_1(pool_1)
        contr_2_2 = self.contr_2_2(contr_2_1)
        pool_2 = self.pool_2(contr_2_2)

        contr_3_1 = self.contr_3_1(pool_2)
        contr_3_2 = self.contr_3_2(contr_3_1)
        pool_3 = self.pool_3(contr_3_2)

        if self.dropout:
            pool_3 = self.dropout(pool_3)

        encode_1 = self.encode_1(pool_3)
        encode_2 = self.encode_2(encode_1)
        deconv_1 = self.deconv_1(encode_2)

        concat2 = torch.cat([deconv_1, contr_3_2], 1)
        expand_2_1 = self.expand_2_1(concat2)
        expand_2_2 = self.expand_2_2(expand_2_1)
        deconv_3 = self.deconv_3(expand_2_2)

        output_2 = self.output_2(concat2)
        output_2_up = self.output_2_up(output_2)

        concat3 = torch.cat([deconv_3, contr_2_2], 1)
        expand_3_1 = self.expand_3_1(concat3)
        expand_3_2 = self.expand_3_2(expand_3_1)
        deconv_4 = self.deconv_4(expand_3_2)

        output_3 = output_2_up + self.output_3(concat3)
        output_3_up = self.output_3_up(output_3)

        concat4 = torch.cat([deconv_4, contr_1_2], 1)
        expand_4_1 = self.expand_4_1(concat4)
        expand_4_2 = self.expand_4_2(expand_4_1)

        conv_5 = self.conv_5(expand_4_2)

        final = output_3_up + conv_5

        return final