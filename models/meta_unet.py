
""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.ops import *

class ConvD(nn.Module):
    def __init__(self, inplanes, planes, norm='in', first=False):
        super(ConvD, self).__init__()

        self.first = first

        self.conv1 = nn.Conv2d(inplanes, planes, 3, 1, 1, bias=True)
        self.in1   = normalization(planes, norm)

        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=True)
        self.in2   = normalization(planes, norm)

        self.conv3 = nn.Conv2d(planes, planes, 3, 1, 1, bias=True)
        self.in3   = normalization(planes, norm)

    def forward(self, x, meta_loss, meta_step_size, stop_gradient,):

        self.meta_loss = meta_loss
        self.meta_step_size = meta_step_size
        self.stop_gradient = stop_gradient

        if not self.first:
            x = maxpool2D(x, kernel_size=2)

        #layer 1 conv, in
        x = conv2d(x, self.conv1.weight, self.conv1.bias, stride=1, padding=1, meta_loss=self.meta_loss, meta_step_size=self.meta_step_size, stop_gradient=self.stop_gradient)
        x = self.in1(x)

        #layer 2 conv, in, lrelu
        y = conv2d(x, self.conv2.weight, self.conv2.bias, stride=1, padding=1, meta_loss=self.meta_loss, meta_step_size=self.meta_step_size, stop_gradient=self.stop_gradient)
        y = self.in2(y)
        y = lrelu(y)

        #layer 3 conv, in, lrelu
        z = conv2d(y, self.conv3.weight, self.conv3.bias, stride=1, padding=1, meta_loss=self.meta_loss, meta_step_size=self.meta_step_size, stop_gradient=self.stop_gradient)
        z = self.in3(z)
        z = lrelu(z)

        return z

class ConvU(nn.Module):
    def __init__(self, planes, norm='in', first=False):
        super(ConvU, self).__init__()

        self.first = first
        if not self.first:
            self.conv1 = nn.Conv2d(2*planes, planes, 3, 1, 1, bias=True)
            self.in1   = normalization(planes, norm)

        self.conv2 = nn.Conv2d(planes, planes//2, 1, 1, 0, bias=True)
        self.in2   = normalization(planes//2, norm)

        self.conv3 = nn.Conv2d(planes, planes, 3, 1, 1, bias=True)
        self.in3   = normalization(planes, norm)

    def forward(self, x, prev, meta_loss, meta_step_size, stop_gradient,):

        self.meta_loss = meta_loss
        self.meta_step_size = meta_step_size
        self.stop_gradient = stop_gradient

        #layer 1 conv, in, lrelu
        if not self.first:
            x = conv2d(x, self.conv1.weight, self.conv1.bias, stride=1, padding=1, meta_loss=self.meta_loss, meta_step_size=self.meta_step_size, stop_gradient=self.stop_gradient)
            x = self.in1(x)
            x = lrelu(x)

        #upsample, layer 2 conv, bn, relu
        y = upsample(x)
        y = conv2d(y, self.conv2.weight, self.conv2.bias, stride=1, padding=0, meta_loss=self.meta_loss, meta_step_size=self.meta_step_size, stop_gradient=self.stop_gradient)
        y = self.in2(y)
        y = lrelu(y)

        #concatenation of two layers
        y = torch.cat([prev, y], 1)

        #layer 3 conv, bn
        y = conv2d(y, self.conv3.weight, self.conv3.bias, stride=1, padding=1, meta_loss=self.meta_loss, meta_step_size=self.meta_step_size, stop_gradient=self.stop_gradient)
        y = self.in3(y)
        y = lrelu(y)

        return y

class UNet(nn.Module):
    def __init__(self, c, n, num_classes, norm='in'):
        super(UNet, self).__init__()

        self.convd1 = ConvD(c,     n, norm, first=True)
        self.convd2 = ConvD(n,   2*n, norm)
        self.convd3 = ConvD(2*n, 4*n, norm)
        self.convd4 = ConvD(4*n, 8*n, norm)
        self.convd5 = ConvD(8*n,16*n, norm)

        self.convu4 = ConvU(16*n, norm, first=True)
        self.convu3 = ConvU(8*n, norm)
        self.convu2 = ConvU(4*n, norm)
        self.convu1 = ConvU(2*n, norm)

        self.seg1 = nn.Conv2d(2*n, num_classes, 1)

    def forward(self, x, meta_loss, meta_step_size, stop_gradient):
        self.meta_loss = meta_loss
        self.meta_step_size = meta_step_size
        self.stop_gradient = stop_gradient

        x1 = self.convd1(x, meta_loss=self.meta_loss, meta_step_size=self.meta_step_size,
                         stop_gradient=self.stop_gradient)
        x2 = self.convd2(x1, meta_loss=self.meta_loss, meta_step_size=self.meta_step_size,
                         stop_gradient=self.stop_gradient)
        x3 = self.convd3(x2, meta_loss=self.meta_loss, meta_step_size=self.meta_step_size,
                         stop_gradient=self.stop_gradient)
        x4 = self.convd4(x3, meta_loss=self.meta_loss, meta_step_size=self.meta_step_size,
                         stop_gradient=self.stop_gradient)
        x5 = self.convd5(x4, meta_loss=self.meta_loss, meta_step_size=self.meta_step_size,
                         stop_gradient=self.stop_gradient)

        y4 = self.convu4(x5, x4, meta_loss=self.meta_loss, meta_step_size=self.meta_step_size,
                         stop_gradient=self.stop_gradient)
        y3 = self.convu3(y4, x3, meta_loss=self.meta_loss, meta_step_size=self.meta_step_size,
                         stop_gradient=self.stop_gradient)
        y2 = self.convu2(y3, x2, meta_loss=self.meta_loss, meta_step_size=self.meta_step_size,
                         stop_gradient=self.stop_gradient)
        y1 = self.convu1(y2, x1, meta_loss=self.meta_loss, meta_step_size=self.meta_step_size,
                         stop_gradient=self.stop_gradient)

        y1 = conv2d(y1, self.seg1.weight, self.seg1.bias, meta_loss=self.meta_loss, meta_step_size=self.meta_step_size, stop_gradient=self.stop_gradient, kernel_size=None, stride=1, padding=0)

        return y1