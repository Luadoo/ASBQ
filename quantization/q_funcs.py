import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Function
" source code: https://github.com/zzzxxxttt/pytorch_DoReFaNet/blob/master/utils/quant_dorefa.py "


# def curve_relu6(x):
#   new_sign = torch.clamp(F.relu6(torch.tanh(x * 2) + 0.5), max=6)
#   return


def uniform_quantize(k):
  class qfn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
      if k == 32:
        out = input
      elif k == 1:
        sign = torch.sign(input)
        out = sign * torch.floor(torch.abs(input) + 0.5)
       # out = sign * (torch.floor(torch.abs(input) + 0.5) ** (1 / 2))
      # elif k == 2:
      #   sign = torch.sign(input)
      #   out = sign * torch.floor(torch.abs(input) + 0.5)
      # elif k == 4:
      #   sign = torch.sign(input)
      #   out = sign * torch.floor(torch.abs(input) + 0.5)
      # elif k == 8:
      #   sign = torch.sign(input)
      #   out = sign * torch.floor(torch.abs(input) + 0.5)

      else:
        n = float(2 ** k - 1)
        out = torch.round(input * n) / n
      return out

    @staticmethod
    # def backward(ctx, grad_output):
    #     input, = ctx.saved_tensors
    #     grad_Htanh = grad_output.clone()
    #     grad_Htanh[input.abs() > 1] = 0
    #     return grad_Htanh
    def backward(ctx, grad_output):
      grad_input = grad_output.clone()
      return grad_input

  return qfn().apply


class weight_quantize_fn(nn.Module):
  def __init__(self, w_bit):
    super(weight_quantize_fn, self).__init__()
    self.w_bit = w_bit    ##
    self.uniform_q = uniform_quantize(k=w_bit)   ##
    assert w_bit <= 8 or w_bit == 32  ##
  # def round(self, x):
  #   weight_q = Round.apply(x)
  #   return weight_q

  def forward(self, x):

    if self.w_bit == 32:
      weight_q = x
    elif self.w_bit == 1:
      E = torch.mean(torch.abs(x)).detach()        ##
   ##   weight_q = self.uniform_q(torch.clip(x, -1, 1))
   ##   weight_q = weight_q * E
      weight_q = self.uniform_q(x / E) * E         ##
      #print(weight_q)
    # elif self.w_bit == 2:
    #   E = torch.mean(torch.abs(x)).detach()  ##
    #   weight_q = self.uniform_q(x / E) * E  ##
    # elif self.w_bit == 4:
    #   E = torch.mean(torch.abs(x)).detach()  ##
    #   weight_q = self.uniform_q(x / E) * E  ##
    # elif self.w_bit == 8:
    #   E = torch.mean(torch.abs(x)).detach()  ##
    #   weight_q = self.uniform_q(x / E) * E  ##
    else:
     # weight = curve_relu6(x)
      weight = torch.tanh(x)
    ##  weight_clip = torch.clip(x, -1, 1)
      max_w = torch.max(torch.abs(weight)).detach()
      weight = weight / 2 / max_w + 0.5
      weight_q = max_w * (2 * self.uniform_q(weight) - 1)

    return weight_q


class activation_quantize_fn(nn.Module):
  def __init__(self, a_bit):
    super(activation_quantize_fn, self).__init__()
    assert a_bit <= 8 or a_bit == 32
    self.a_bit = a_bit
    self.uniform_q = uniform_quantize(k=a_bit)

  def forward(self, x):
    if self.a_bit == 32:
      activation_q = x
    else:
      activation_q = self.uniform_q(torch.clamp(x, 0, 1))
      # print(np.unique(activation_q.detach().numpy()))
    return activation_q


def conv2d_Q_fn(w_bit):
  class Conv2d_Q(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
      super(Conv2d_Q, self).__init__(in_channels, out_channels, kernel_size, stride,
                                     padding, dilation, groups, bias)
      self.w_bit = w_bit
      self.quantize_fn = weight_quantize_fn(w_bit=w_bit)

    def forward(self, input, order=None):
      weight_q = self.quantize_fn(self.weight)
      print(weight_q)
      # print(np.unique(weight_q.detach().numpy()))
      return F.conv2d(input, weight_q, self.bias, self.stride,
                      self.padding, self.dilation, self.groups)

  return Conv2d_Q


def linear_Q_fn(w_bit):
  class Linear_Q(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
      super(Linear_Q, self).__init__(in_features, out_features, bias)
      self.w_bit = w_bit
      self.quantize_fn = weight_quantize_fn(w_bit=w_bit)

    def forward(self, input):
      weight_q = self.quantize_fn(self.weight)
      # print(np.unique(weight_q.detach().numpy()))
      return F.linear(input, weight_q, self.bias)

  return Linear_Q

# batchnorm_fn = batchnorm2d_fn()
