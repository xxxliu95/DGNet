import torch
import torch.nn as nn
import torch.nn.functional as F
from models.ops import *

class StyleEncoder(nn.Module):
    def __init__(self, style_dim):
        super(StyleEncoder, self).__init__()
        dim = 8
        self.style_dim = style_dim // 2
        self.conv1 = nn.Conv2d(1, dim, 7, 1, 3, bias=True)
        self.conv2 = nn.Conv2d(dim, dim*2, 4, 2, 1, bias=True)
        self.conv3 = nn.Conv2d(dim*2, dim*4, 4, 2, 1, bias=True)
        self.conv4 = nn.Conv2d(dim*4, dim*8, 4, 2, 1, bias=True)
        self.conv5 = nn.Conv2d(dim*8, dim*16, 4, 2, 1, bias=True)
        self.conv6 = nn.Conv2d(dim*16, dim*32, 4, 2, 1, bias=True)

        self.fc1 = nn.Linear(256*9*9, 4*9*9)
        self.fc2 = nn.Linear(4*9*9, 32)
        self.mu = nn.Linear(32, style_dim)
        self.logvar = nn.Linear(32, style_dim)
        self.classifier = nn.Linear(self.style_dim, 3)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mu + eps * std

    def forward(self, x, meta_loss, meta_step_size, stop_gradient):
        self.meta_loss = meta_loss
        self.meta_step_size = meta_step_size
        self.stop_gradient = stop_gradient

        out = conv2d(x, self.conv1.weight, self.conv1.bias, stride=1, padding=3, meta_loss=self.meta_loss, meta_step_size=self.meta_step_size,
               stop_gradient=self.stop_gradient)
        out = conv2d(out, self.conv2.weight, self.conv2.bias, stride=2, padding=1, meta_loss=self.meta_loss, meta_step_size=self.meta_step_size,
               stop_gradient=self.stop_gradient)
        out = conv2d(out, self.conv3.weight, self.conv3.bias, stride=2, padding=1, meta_loss=self.meta_loss, meta_step_size=self.meta_step_size,
               stop_gradient=self.stop_gradient)
        out = conv2d(out, self.conv4.weight, self.conv4.bias, stride=2, padding=1, meta_loss=self.meta_loss, meta_step_size=self.meta_step_size,
               stop_gradient=self.stop_gradient)
        out = conv2d(out, self.conv5.weight, self.conv5.bias, stride=2, padding=1, meta_loss=self.meta_loss, meta_step_size=self.meta_step_size,
               stop_gradient=self.stop_gradient)
        out = conv2d(out, self.conv6.weight, self.conv6.bias, stride=2, padding=1, meta_loss=self.meta_loss, meta_step_size=self.meta_step_size,
               stop_gradient=self.stop_gradient)

        out = linear(out.view(-1, out.shape[1] * out.shape[2] * out.shape[3]), self.fc1.weight, self.fc1.bias, meta_loss=self.meta_loss,
                     meta_step_size=self.meta_step_size, stop_gradient=self.stop_gradient)
        out = lrelu(out)
        out = linear(out, self.fc2.weight, self.fc2.bias, meta_loss=self.meta_loss, meta_step_size=self.meta_step_size,
               stop_gradient=self.stop_gradient)
        out = lrelu(out)

        mu = linear(out, self.mu.weight, self.mu.bias, meta_loss=self.meta_loss, meta_step_size=self.meta_step_size,
                     stop_gradient=self.stop_gradient)
        logvar = linear(out, self.logvar.weight, self.logvar.bias, meta_loss=self.meta_loss, meta_step_size=self.meta_step_size,
                     stop_gradient=self.stop_gradient)

        zs = self.reparameterize(mu[:,self.style_dim:], logvar[:,self.style_dim:])
        zd = self.reparameterize(mu[:,:self.style_dim], logvar[:,:self.style_dim])
        z = torch.cat((zs,zd), dim=1)

        cls = linear(z[:,:self.style_dim], self.classifier.weight, self.classifier.bias, meta_loss=self.meta_loss, meta_step_size=self.meta_step_size,
                     stop_gradient=self.stop_gradient)
        cls = F.softmax(cls, dim=1)
        return z, mu, logvar, cls