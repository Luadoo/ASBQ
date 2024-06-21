
import resnet_quant
import argparse
import torch
import torch.nn as nn
from torch.backends import cudnn
#from quantization.q_funcs import *

import torch.nn.parallel
import torch.distributed as dist
import os
#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# def str2bool(v):
#     if v.lower() in ('yes', 'true', 't', 'y', '1'):
#         return True
#     else:
#         return False

def str2gpu(v):
    try:
        gpu_id = int(v)
        if gpu_id < 0:
            print("invalid GPU ID. please use a non-negative integer.")
            exit(1)
        return gpu_id
    except ValueError:
        print("Invalid GPU value. please use a non-negative integer.")
        exit(1)


def parse_arguments():
    parser = argparse.ArgumentParser(description='TA Knowledge Distillation Code')
    parser.add_argument('--epochs', default=100, type=int,  help='number of total epochs to run')
    parser.add_argument('--dataset', default='imagenet', type=str, help='dataset. can be cifar10')
    parser.add_argument('--batch-size', default=512, type=int, help='batch_size')
    parser.add_argument('--learning-rate', default=0.1, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,  help='SGD momentum')
  #  parser.add_argument('--dampening', default=0.5, type=float, help='SGD dampening')
    parser.add_argument('--weight-decay', default=1e-4, type=float, help='SGD weight decay (default: 5e-4)')
    parser.add_argument('--teacher', default='', type=str, help='teacher student name')
    parser.add_argument('--student', '--model', default='resnet20', type=str, help='teacher student name')
    parser.add_argument('--teacher-checkpoint', default='', type=str, help='optional pretrained checkpoint for teacher')
    parser.add_argument('--cuda', default=-1, type=str2gpu, help='whether or not use cuda(train on GPU)')
  #  parser.add_argument('--cuda', default=-1, type=int, help='GPU device to use (default: -1 for CPU)')
    parser.add_argument('--dataset-dir', default='./data', type=str,  help='dataset directory')
    parser.add_argument('--trial-id', default='[trial_id_not_used]', type=str,  help='dataset directory')
    parser.add_argument('--test-run', action='store_true', help='test everything loads correctly')
    parser.add_argument('--teacher-wbits', type=int, default=32)
    parser.add_argument('--teacher-abits', type=int, default=32)
    parser.add_argument('--teacher-quantization', default='dorefa')
    parser.add_argument('--student-wbits', type=int, default='1', help='student model weight bit-widths')
    parser.add_argument('--student-abits', type=int, default='1', help='student model activation bit-widths')
    parser.add_argument('--student-quantization', default='dorefa')
    parser.add_argument('--seed', type=int, default=1000)

    # parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
    # parser.add_argument('--kd_type', default='', help='label smoothing')
    args = parser.parse_args()
    # if args.cuda >= 0:
    #     torch.cuda.set_device(args.cuda)
    return args

def main():
    args = parse_arguments()

    if args.cuda >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
        cudnn.benchmark = True

    model = resnet_quant.resnet_models()
    
    if args.cuda >= 0:
        #torch.nn.parallel.DistributedDataParallel(resnet_models)
      #  torch.cuda.set_device(args.cuda)                    ##
        model = torch.nn.DataParallel(model)
      #  model = model.cuda()
        model = model.to(device)
      #  torch.save(model.module.state_dict(), 'model.pth.tar')

    if args.teacher_checkpoint:
        model = load_checkpoint(model, args.teacher_checkpoint)

    else:
        torch.device('cuda')
       # torch.nn.DataParallel(resnet_models)
      #  print(' it is error!')
    return args

#
def load_checkpoint(model, checkpoint_path):
    """
    Loads weights from checkpoint
    :param model: a pytorch nn student
    :param str checkpoint_path: address/path of a file
    :return: pytorch nn student with weights loaded from checkpoint
    """

    model_ckp = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model_ckp = model_ckp['model_state_dict']
    # model_ckp = torch.load(checkpoint_path)
    # model.load_state_dict(model_ckp['model_state_dict'])

    new_state = {}
    for key in model.state_dict().keys():
        new_state[key] = model_ckp[key]
    model.load_state_dict(new_state)

    return model


if __name__ == "__main__":
    main()
