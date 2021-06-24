import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.autograd import Variable

def normalization(planes, norm='in'):
    if norm == 'bn':
        m = nn.BatchNorm2d(planes)
    elif norm == 'gn':
        m = nn.GroupNorm(1, planes)
    elif norm == 'in':
        m = nn.InstanceNorm2d(planes)
    else:
        raise ValueError('normalization type {} is not supported'.format(norm))
    return m

def linear(inputs, weight, bias, meta_step_size=0.001, meta_loss=None, stop_gradient=False):
    inputs = inputs
    weight = weight
    bias = bias

    if meta_loss is not None:

        if not stop_gradient:
            grad_weight = autograd.grad(meta_loss, weight, create_graph=True, allow_unused=True) [0]

            if bias is not None:
                grad_bias = autograd.grad(meta_loss, bias, create_graph=True, allow_unused=True) [0]
                bias_adapt = bias - grad_bias * meta_step_size
            else:
                bias_adapt = bias

        else:
            grad_weight = Variable(autograd.grad(meta_loss, weight, create_graph=True, allow_unused=True)[0].data, requires_grad=False)

            if bias is not None:
                grad_bias = Variable(autograd.grad(meta_loss, bias, create_graph=True, allow_unused=True)[0].data, requires_grad=False)
                bias_adapt = bias - grad_bias * meta_step_size
            else:
                bias_adapt = bias

        return F.linear(inputs,
                        weight - grad_weight * meta_step_size,
                        bias_adapt)
    else:
        return F.linear(inputs, weight, bias)


def conv2d(inputs, weight, bias, meta_step_size=0.001, stride=1, padding=0, dilation=1, groups=1, meta_loss=None,
           stop_gradient=False, kernel_size=None):

    inputs = inputs
    weight = weight
    bias = bias


    if meta_loss is not None:

        if not stop_gradient:
            grad_weight = autograd.grad(meta_loss, weight, create_graph=True, allow_unused=True)[0]

            if bias is not None:
                grad_bias = autograd.grad(meta_loss, bias, create_graph=True, allow_unused=True)[0]
                if grad_bias is not None:
                    bias_adapt = bias - grad_bias * meta_step_size
                else:
                    bias_adapt = bias
            else:
                bias_adapt = bias

        else:
            grad_weight = Variable(autograd.grad(meta_loss, weight, create_graph=True, allow_unused=True)[0].data,
                                   requires_grad=False)
            if bias is not None:
                grad_bias = Variable(autograd.grad(meta_loss, bias, create_graph=True, allow_unused=True)[0].data, requires_grad=False)
                bias_adapt = bias - grad_bias * meta_step_size
            else:
                bias_adapt = bias
        if grad_weight is not None:
            weight_adapt = weight - grad_weight * meta_step_size
        else:
            weight_adapt = weight

        return F.conv2d(inputs,
                        weight_adapt,
                        bias_adapt, stride,
                        padding,
                        dilation, groups)
    else:
        return F.conv2d(inputs, weight, bias, stride, padding, dilation, groups)


def deconv2d(inputs, weight, bias, meta_step_size=0.001, stride=2, padding=0, dilation=0, groups=1, meta_loss=None,
           stop_gradient=False, kernel_size=None):

    inputs = inputs
    weight = weight
    bias = bias


    if meta_loss is not None:

        if not stop_gradient:
            grad_weight = autograd.grad(meta_loss, weight, create_graph=True, allow_unused=True)[0]

            if bias is not None:
                grad_bias = autograd.grad(meta_loss, bias, create_graph=True, allow_unused=True)[0]
                bias_adapt = bias - grad_bias * meta_step_size
            else:
                bias_adapt = bias

        else:
            grad_weight = Variable(autograd.grad(meta_loss, weight, create_graph=True, allow_unused=True)[0].data,
                                   requires_grad=False)
            if bias is not None:
                grad_bias = Variable(autograd.grad(meta_loss, bias, create_graph=True, allow_unused=True)[0].data, requires_grad=False)
                bias_adapt = bias - grad_bias * meta_step_size
            else:
                bias_adapt = bias

        return F.conv_transpose2d(inputs,
                        weight - grad_weight * meta_step_size,
                        bias_adapt, stride,
                        padding,
                        dilation, groups)
    else:
        return F.conv_transpose2d(inputs, weight, bias, stride, padding, dilation, groups)

def tanh(inputs):
    return torch.tanh(inputs)

def relu(inputs):
    return F.relu(inputs, inplace=True)

def lrelu(inputs):
    return F.leaky_relu(inputs, negative_slope=0.01, inplace=False)

def maxpool(inputs, kernel_size, stride=None, padding=0):
    return F.max_pool2d(inputs, kernel_size, stride, padding=padding)

def dropout(inputs):
    return F.dropout(inputs, p=0.5, training=False, inplace=False)

def batchnorm(inputs, running_mean, running_var):
    return F.batch_norm(inputs, running_mean, running_var)

def instancenorm(input):
    return F.instance_norm(input)

def groupnorm(input):
    return F.group_norm(input)

def dropout2D(inputs):
    return F.dropout2d(inputs, p=0.5, training=False, inplace=False)

def maxpool2D(inputs, kernel_size, stride=None, padding=0):
    return F.max_pool2d(inputs, kernel_size, stride, padding=padding)

def upsample(input):
    return F.interpolate(input, scale_factor=2, mode='bilinear', align_corners=False)
