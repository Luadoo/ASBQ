"""
resnet for cifar in pytorch

Reference:
[1] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In CVPR, 2016.
[2] K. He, X. Zhang, S. Ren, and J. Sun. Identity mappings in deep residual networks. In ECCV, 2016.
"""
import torch
import torch.nn as nn
import math

from torchvision.models.quantization import resnet18

from quantization.q_funcs import *
from utils.dataset_loader import get_dataset
from torch.nn import PReLU
import torch.nn.functional as F
from torch.nn.utils import prune
# import torch.nn.init as init
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from ops import SyncSwitchableNorm2d, SwitchNorm2d, SyncBatchNorm2d


class Conv2d_Q(nn.Conv2d):

    def __init__(self, wbit, in_planes, out_planes, kernel_size=3, stride=1,
                 padding=1, q_method='dorefa', bias=False, dropout_rate=0.05):

        super(Conv2d_Q, self).__init__(in_planes, out_planes, kernel_size, stride,
                                       padding, bias=False)
        self.qfn = weight_quantize_fn(w_bit=wbit)
        self.dropout_rate = dropout_rate

        if self.dropout_rate is not None:
            prune.ln_structured(self, name='weight', amount=self.dropout_rate, n=2, dim=0)
         #   prune.l1_unstructured(self, name='weight', amount=self.dropout_rate)

    def forward(self, x):
        weights_q = self.qfn(self.weight)
        out = nn.functional.conv2d(x, weights_q, self.bias, self.stride,
                                    self.padding)
        return out

'''
# @BL bug with setting weights_q and weights_fp when loading
class Conv2d_Q(nn.Conv2d):

    def __init__(self, wbit, in_planes, out_planes, kernel_size=3, stride=1,
                 padding=1, q_method='dorefa', bias=False):

        super(Conv2d_Q, self).__init__(in_planes, out_planes, kernel_size, stride,
                                       padding, bias=False)
        self.qfn = weight_quantize_fn(w_bit=wbit)


    def forward(self, x):
        weights_q = self.qfn(self.weight)
        self.weights_q = weights_q
        self.weights_fp = self.weight
        return nn.functional.conv2d(x, weights_q, self.bias, self.stride,
                                    self.padding)
'''

# class GeLU(nn.Module):
#     def __init__(self):
#         super(GeLU, self).__init__()
#     def forward(self, x):
#         return 0.5 * x * (1 + torch.tanh(math.sqrt(2/math.pi) * (x+0.044715 * torch.pow(x,3))))
#



class PreActBasicBlock_convQ(nn.Module):
    expansion = 1

    def __init__(self, q_method, wbit, abit, in_planes, out_planes, stride=1, downsample=None):
        super(PreActBasicBlock_convQ, self).__init__()
        self.act_qfn = activation_quantize_fn(a_bit=abit)

        self.relu1 = nn.ReLU6(inplace=True)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = Conv2d_Q(wbit, in_planes, out_planes, stride=stride, kernel_size=3, padding=1, bias=False,
                              q_method=q_method)

        # TODO: Move class from this file
    #    init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        self.relu2 = nn.ReLU6(inplace=True)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = Conv2d_Q(wbit, out_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False,
                              q_method=q_method)

     #   init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')

        self.downsample = downsample
       # self.prelu = nn.PReLU()    # resourse
      #  self.tanh = nn.Hardtanh(inplace=True)
        self.stride = stride
       ## self.dropout = nn.Dropout(p=dropout_rate)
        ######
        # if stride != 1:
        #     self.skip_conv = Conv2d_Q(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)
        #     self.skip_bn = nn.BatchNorm2d(out_planes)
        # ######
    def forward(self, x):
        residual = x


        # activation qunatization applied here
        out = self.act_qfn(self.relu1(x))
        out = self.bn1(out)

        # TODO: check how residual is accounted for. DoReFa seems to leave residual full precision??
        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.conv1(out)


        out = self.act_qfn(self.relu2(out))
        out = self.bn2(out)
        out = self.conv2(out)


        out += residual

      #  out = torch.clamp(F.relu6(torch.tanh(out * 2) + 0.5), max=1)

     ##   out = self.dropout(out)
       # out = self.relu(out)
        return out


class ResNet_imagenet(nn.Module):
    def __init__(self, block, layers, wbit, abit, num_classes=1000, q_method='dorefa'):
        super(ResNet_imagenet, self).__init__()
        self.inplanes = 64


        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.relu = nn.ReLU6(inplace=True)  # only deplay on imagenet in here
        self.bn1 = nn.BatchNorm2d(64)  # only deplay on imagenet in here


        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


        # if layers == 18:
        #     layers_config = [2, 2, 2, 2]
        #     block = PreActBasicBlock_convQ
        # elif layers == 34:
        #     layers_config = [3, 4, 6, 3]
        #     block = PreActBasicBlock_convQ
        # elif layers == 50:
        #     layers_config = [3, 4, 6, 3]
        #     block = PreActBottleneck_Q
        # else:
        #     print('Invalid network depth')
        #     exit()

        self.layer1 = self._make_layer(block, 64, wbit, abit, layers[0], stride=1, q_method=q_method)
        self.layer2 = self._make_layer(block, 128, wbit, abit, layers[1], stride=2, q_method=q_method)
        self.layer3 = self._make_layer(block, 256, wbit, abit, layers[2], stride=2, q_method=q_method)
        self.layer4 = self._make_layer(block, 512, wbit, abit, layers[3], stride=2, q_method=q_method)
      #  self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    #    self.bn1 = nn.BatchNorm2d(512 * block.expansion)     # only for cifar10,100 in here

     #   if self.include_top:
        self.avgpool = nn.AvgPool2d(7, stride=1)
        #self.avgpool = nn.AvgPool2d(4, stride=1)     # only deplay on cifar10,100 in here
    #    self.bn2 = nn.BatchNorm2d(512 * block.expansion)
        self.fc = nn.Linear(512 * block.expansion, 1000)

       # self.bn2 = nn.BatchNorm1d(layers[3]*block.expansion)
   ##     self.apply(_weights_init)

    #    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    #
    #    for m in self.modules():
    #        if isinstance(m, nn.Conv2d):
    #            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #            m.weight.data.normal_(0, math.sqrt(2. / n))
    #        elif isinstance(m, nn.BatchNorm2d):
    #            if m.weight is not None:
    #                m.weight.data.fill_(1)
    #            if m.bias is not None:
    #                m.bias.data.zero_()


#  behind just for cifar10,100

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        #
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight)

    def _make_layer(self, block, planes, wbit, abit, blocks, stride=1, q_method='dorefa'):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )
        layers = []
        layers.append(block(q_method, wbit, abit, self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(q_method, wbit, abit, self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)  # only deplay on imagenet
        x = self.bn1(x)  # only deplay on imagenet


        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)


      #  if self.include_top:
      #   x = self.bn1(x)
      #   x = self.relu(x)
    #    x = self.maxpool(x)

      #  x = F.avg_pool2d(x, x.size()[3])

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
      #  x = self.bn2(x)
     #   x = torch.flatten(x, 1)
        x = self.fc(x)
        return x



def resnet18_imagenet(wbit, abit, q_method=None, **kwargs):
    model = ResNet_imagenet(PreActBasicBlock_convQ, [2, 2, 2, 2], wbit, abit, q_method=q_method, **kwargs)
    return model


def resnet34_cifar(wbit, abit, q_method=None, **kwargs):
    model = ResNet_imagenet(PreActBasicBlock_convQ, [3, 4, 6, 3], wbit, abit, q_method=q_method, **kwargs)
    return model


resnet_models = {

    '18': resnet18_imagenet,
    '34': resnet34_cifar,

}

def is_resnet(name):
    """
    Simply checks if name represents a resnet, by convention, all resnet names start with 'resnet'
    :param name:
    :return:
    """
    name = name.lower()
    return name.startswith('resnet')

def get_quant_model(name, qparams, dataset="imagenet", use_cuda=False):
    """
    Create a student for training, given student name and dataset
    :param name: name of the student. e.g., resnet110, resnet32, plane2, plane10, ...
    :param dataset: the dataset which is used to determine last layer's output size. Options are cifar10 and cifar100.
    :return: a pytorch student for neural network
    """
    num_classes = 1000 if dataset == 'imagenet' else 1000
    wbits, abits, q_method = qparams

    model = None
    if is_resnet(name):
        resnet_size = name[6:]
        resnet_model = resnet_models.get(resnet_size)
        model = resnet_model(wbits, abits, q_method, num_classes=num_classes)
    else:
        raise Exception('not resnet!')

    # copy to cuda if activated
    if use_cuda:
        model = model.cuda()
    return model
