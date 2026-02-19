import torch
import torch.nn as nn
import math


from quantization.q_funcs import *
from utils.dataset_loader import get_dataset
from torch.nn import PReLU
import torch.nn.functional as F
from torch.nn.utils import prune
# import torch.nn.init as init
#import matplotlib.pyplot as plt
#from PIL import Image
from torchvision import transforms


# class Structured1xNPruning(prune.BasePruningMethod):
#     PRUNING_TYPE = 'structured'

#     def __init__(self, N, dims):
#         self.N = N
#         self.dims = dims  # dims是一个列表，例如[2, 3]

#     def compute_mask(self, t, default_mask):
#         mask = default_mask.clone()
#         for dim in self.dims:
#             if dim == 3:
#                 for out_c in range(t.size(0)):
#                     for in_c in range(t.size(1)):
#                         for h in range(t.size(2)):
#                             for w in range(0, t.size(3), self.N):
#                                 if w + self.N <= t.size(3):
#                                     block = t[out_c, in_c, h, w:w+self.N]
#                                     norm = torch.norm(block, p=1).item()
#                                     threshold = 1e-3
#                                     if norm < threshold:
#                                         mask[out_c, in_c, h, w:w+self.N] = 0
#             elif dim == 2:
#                 for out_c in range(t.size(0)):
#                     for in_c in range(t.size(1)):
#                         for w in range(t.size(3)):
#                             for h in range(0, t.size(2), self.N):
#                                 if h + self.N <= t.size(2):
#                                     block = t[out_c, in_c, h:h+self.N, w]
#                                     norm = torch.norm(block, p=1).item()
#                                     threshold = 1e-3
#                                     if norm < threshold:
#                                         mask[out_c, in_c, h:h+self.N, w] = 0
#             else:
#                 raise NotImplementedError(f"仅支持dim=3和dim=2方向的1xN剪枝，目前dim={dim}不支持。")
#         return mask

#     @staticmethod
#     def apply_pruning(module, name, N, dims):
#         # 直接实例化 Structured1xNPruning 并传入 N 和 dims 参数
#         pruning_method = Structured1xNPruning(N, dims)
#         prune.custom_from_mask(module, name, pruning_method.compute_mask(module.weight, torch.ones_like(module.weight)))


# class Conv2d_Q(nn.Conv2d):
#     def __init__(self, wbit, in_planes, out_planes, kernel_size=3, stride=1,
#                  padding=1, q_method='dorefa', bias=False, dropout_rate=0.05, N=2, dim=2):
#         super(Conv2d_Q, self).__init__(in_planes, out_planes, kernel_size, stride,
#                                        padding, bias=False)
#         self.qfn = weight_quantize_fn(w_bit=wbit)  # 假设量化函数已定义
#         self.dropout_rate = dropout_rate
#         self.N = N
#         self.dim = [dim] if isinstance(dim, int) else dim  # 修改为支持多个维度

#         if self.dropout_rate is not None:

#             Structured1xNPruning.apply_pruning(self, 'weight', self.N, self.dim)
         
#     def forward(self, x):
#         # 在前向传播中应用剪枝掩码和量化权重
#         if hasattr(self.weight, 'mask'):
#             weights_q = self.qfn(self.weight * self.weight.mask)
#         else:
#             weights_q = self.qfn(self.weight)
#         return nn.functional.conv2d(x, weights_q, self.bias, self.stride, self.padding)

    #定义1xN结构化剪枝方法
class Structured1xNPruning(prune.BasePruningMethod):
    PRUNING_TYPE = 'structured'

    def __init__(self, N, dim):
        self.N = N
        self.dim = dim

    def compute_mask(self, t, default_mask):
        mask = default_mask.clone()
        if self.dim == 3:
            for out_c in range(t.size(0)):
                for in_c in range(t.size(1)):
                    for h in range(t.size(2)):
                        for w in range(0, t.size(3), self.N):
                            if w + self.N <= t.size(3):
                                block = t[out_c, in_c, h, w:w+self.N]
                                norm = torch.norm(block, p=1).item()
                                threshold = 1e-3
                                if norm < threshold:
                                    mask[out_c, in_c, h, w:w+self.N] = 0
        else:
            raise NotImplementedError("仅实现了dim=3方向的1xN剪枝。")
        return mask

#定义支持量化和剪枝的卷积层
class Conv2d_Q(nn.Conv2d):
    def __init__(self, wbit, in_planes, out_planes, kernel_size=3, stride=1,
                 padding=1, q_method='dorefa', bias=False, dropout_rate=0.05, N=2):
        super(Conv2d_Q, self).__init__(in_planes, out_planes, kernel_size, stride,
                                       padding, bias=False)
        self.qfn = weight_quantize_fn(w_bit=wbit)  # 假设量化函数已定义
        self.dropout_rate = dropout_rate
        self.N = N

        if self.dropout_rate is not None:
            # 使用1xN结构化剪枝
            prune.custom_from_mask(
                self,
                name='weight',
                mask=self.generate_1xN_mask(self.weight, self.dropout_rate, self.N)
            )

    def generate_1xN_mask(self, weight, amount, N):
        mask = torch.ones_like(weight)
        total_blocks = (weight.size(0) * weight.size(1) * weight.size(2) *
                        (weight.size(3) // N))
        num_prune = int(total_blocks * amount)

        norms = []
        for out_c in range(weight.size(0)):
            for in_c in range(weight.size(1)):
                for h in range(weight.size(2)):
                    for w in range(0, weight.size(3), N):
                        if w + N <= weight.size(3):
                            block = weight[out_c, in_c, h, w:w+N]
                            norm = torch.norm(block, p=1).item()
                            norms.append(((out_c, in_c, h, w), norm))

        norms.sort(key=lambda x: x[1])

        for i in range(num_prune):
            (out_c, in_c, h, w), _ = norms[i]
            mask[out_c, in_c, h, w:w+N] = 0

        return mask

    def forward(self, x):
        if hasattr(self.weight, 'mask'):
            weights_q = self.qfn(self.weight * self.weight.mask)
        else:
            weights_q = self.qfn(self.weight)
        return nn.functional.conv2d(x, weights_q, self.bias, self.stride, self.padding)




class vgg_small_FP(nn.Module):
    def __init__(self, wbit, abit, num_classes=10, q_method='dorefa'):
        super(vgg_small_FP, self).__init__()
        self.conv0 = Conv2d_Q(wbit, 3, 128, kernel_size=3, padding=1, q_method=q_method, bias=False)
        self.bn0 = nn.BatchNorm2d(128)
        #self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)




        self.conv1 = Conv2d_Q(wbit, 128, 128, kernel_size=3, padding=1, q_method=q_method, bias=False)
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(128)
        #self.relu1 = nn.ReLU6(inplace=True)

        # 例如，在conv2层进行高度方向剪枝
        self.conv2 = Conv2d_Q(wbit, 128, 256, kernel_size=3, padding=1, q_method=q_method, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        #self.relu3 = nn.ReLU6(inplace=True)


        self.conv3 = Conv2d_Q(wbit, 256, 256, kernel_size=3, padding=1, q_method=q_method, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        #self.relu4 = nn.ReLU6(inplace=True)
        #self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)


        self.conv4 = Conv2d_Q(wbit, 256, 512, kernel_size=3, padding=1, q_method=q_method, bias=False)
        self.act_qfn = activation_quantize_fn(a_bit=abit)
        self.relu1 = nn.ReLU6(inplace=True)
        self.bn4 = nn.BatchNorm2d(512)
        #self.relu5 = nn.ReLU6(inplace=True)


        self.conv5 = Conv2d_Q(wbit, 512, 512, kernel_size=3, padding=1, q_method=q_method, bias=False)
        self.bn5 = nn.BatchNorm2d(512)
        #self.relu6 = nn.ReLU6(inplace=True)
        self.fc = nn.Linear(512*4*4, num_classes)
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, Conv2d_Q):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
    
    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.act_qfn(self.relu1(x))


        x = self.conv1(x)
        x = self.pooling(x)
        x = self.bn1(x)
        x = self.act_qfn(self.relu1(x))


        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act_qfn(self.relu1(x))

        x = self.conv3(x)
        x = self.pooling(x)
        x = self.bn3(x)
        x = self.act_qfn(self.relu1(x))

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.act_qfn(self.relu1(x))


        x = self.conv5(x)
        x = self.pooling(x)
        x = self.bn5(x)
        x = self.act_qfn(self.relu1(x))



        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
        
        
def vgg_small(wbit, abit, q_method=None, **kwargs):
    model = vgg_small_FP(wbit, abit, q_method=q_method, **kwargs)
    return model

vgg_models = {
    'vgg': vgg_small
}

def is_vgg(name):
    """
    检查名字是否以 'vgg' 开头。
    """
    name = name.lower()
    return name.startswith('vgg')

def get_quant_model(name, qparams, dataset="cifar10", use_cuda=False):
    """
    根据模型名称和数据集创建一个量化模型。
    参数：
    - name: 模型名称（例如 'resnet18', 'vggsmall' 等）。
    - qparams: 量化参数，包含 (wbits, abits, q_method)。
    - dataset: 数据集名称，默认为 'imagenet'。
    - use_cuda: 是否使用 CUDA。
    - builder: 构建 VGG 所需的构建器。
    返回：PyTorch 量化模型。
    """
    num_classes = 10 if dataset == 'cifar10' else 100
    wbits, abits, q_method = qparams

    model = None
    if is_vgg(name):
        vgg_size = name
        vgg_model = vgg_models.get(vgg_size)
        if vgg_models is None:
            raise ValueError("VGG 模型需要 builder 参数来构建。")
        model = vgg_models(wbits, abits, q_method, num_classes=num_classes)
    else:
        raise Exception('不支持的模型名称！')

    # 如果启用 CUDA，复制模型到 CUDA
    if use_cuda:
        model = model.cuda()
    return model
