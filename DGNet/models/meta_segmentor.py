import torch
import torch.nn as nn
import torch.nn.functional as F
from models.ops import *
from models.meta_unet import *

class ContentEncoder(nn.Module):
    def __init__(self, width, height, ndf, num_output_channels, norm, upsample):
        super(ContentEncoder, self).__init__()
        """
        Build an encoder to extract anatomical information from the image.
        """
        self.width = width
        self.height = height
        self.ndf = ndf
        self.num_output_channels = num_output_channels
        self.norm = norm
        self.upsample = upsample

        self.unet = UNet(c=1, n=32, norm='in', num_classes=self.num_output_channels)

    def forward(self, x, meta_loss, meta_step_size, stop_gradient):
        self.meta_loss = meta_loss
        self.meta_step_size = meta_step_size
        self.stop_gradient = stop_gradient

        out = self.unet(x, meta_loss=self.meta_loss, meta_step_size=self.meta_step_size, stop_gradient=self.stop_gradient)
        return out

class Segmentor(nn.Module):
    def __init__(self, num_output_channels, num_classes):
        super(Segmentor, self).__init__()
        """
        """
        self.num_output_channels = num_output_channels
        self.num_classes = num_classes+1  # check again

        self.conv1 = nn.Conv2d(self.num_output_channels, 64, 3, 1, 1, bias=True)
        self.bn1   = normalization(64, norm='bn')
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.bn2   = normalization(64, norm='bn')
        self.pred = nn.Conv2d(64, self.num_classes, 1, 1, 0)

    def forward(self, x, meta_loss, meta_step_size, stop_gradient):
        self.meta_loss = meta_loss
        self.meta_step_size = meta_step_size
        self.stop_gradient = stop_gradient

        out = conv2d(x, self.conv1.weight, self.conv1.bias, stride=1, padding=1, meta_loss=self.meta_loss, meta_step_size=self.meta_step_size,
               stop_gradient=self.stop_gradient)
        out = self.bn1(out)
        out = relu(out)
        out = conv2d(out, self.conv2.weight, self.conv2.bias, stride=1, padding=1, meta_loss=self.meta_loss, meta_step_size=self.meta_step_size,
               stop_gradient=self.stop_gradient)
        out = self.bn2(out)
        out = relu(out)
        out = conv2d(out, self.pred.weight, self.pred.bias, stride=1, padding=0, meta_loss=self.meta_loss, meta_step_size=self.meta_step_size, stop_gradient=self.stop_gradient)

        return out