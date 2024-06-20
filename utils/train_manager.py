import os
import json
import pandas as pd
import torch
import copy
import logging
import torch.optim as optim
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from utils.setup_manager import parse_arguments, load_checkpoint
from utils.dataset_loader import get_dataset

from distiller_zoo import KD, RKD
from resnet_imagenet import get_quant_model, is_resnet
#from vgg_quant import get_quant_model, is_vgg

# from torch.backends import cudnn
from torch.optim.lr_scheduler import CosineAnnealingLR
import logging
# import distiller_zoo
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp




class TrainManager(object):
    def __init__(self, student, teacher=None, train_loader=None, test_loader=None, train_config={}):
        self.student = student
        self.teacher = teacher
     ##   self.lambda_relation = train_config.get('lambda_relation', 0.5)
        self.have_teacher = bool(self.teacher)
        self.device = train_config['device']
        if self.device == 'cuda':              ###
        #self.student.to(self.device)
           self.student = torch.nn.DataParallel(self.student)      ###
        self.name = train_config['name']
        self.epochs = train_config['epochs']                                  ##
        self.optimizer = optim.SGD(self.student.parameters(),
                                   lr=train_config['learning_rate'],
                                   momentum=train_config['momentum'],
                             #      dampening=train_config['dampening'],     # decrease oscillation
                                   nesterov=True,
                                   weight_decay=train_config['weight_decay'])
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs)     ##
        if self.have_teacher:
            self.teacher.eval()
            self.teacher.train(mode=False)

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = train_config
        self.accuracy_history = []     #### json

 #   train_manager = TrainManager(student, teacher, train_loader, test_loader, train_config={'epochs': 80, })
        # adjust_learning_rate = optim.lr_scheduler.MultiStepLR(self.optimizer, [100, 150, 180], gamma=0.1)

        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # resnet_quant = resnet20_cifar()
        # resnet_quant = resnet_quant.to(device)
        # if self.device == 'cuda':
        #     net = torch.nn.DataParallel(resnet_quant)
        #     cudnn.benchmark = True

    def train(self):
       # global args
        lambda_ = self.config['lambda_student']
        T = self.config['T_student']
        epochs = self.config['epochs']
        trial_id = self.config['trial_id']
     #   distill = args.kd_type != ''

        max_val_acc = 0
        iteration = 0
        best_acc = 0
        criterion = nn.CrossEntropyLoss()

        train_logs = []    ## 用于存储每个 epoch 的训练记录

        for epoch in range(epochs):
            self.student.train()
            self.scheduler.step()                      ##
            self.adjust_learning_rate(self.optimizer, epoch)
            loss = 0
            total_batches = len(self.train_loader)
            for batch_idx, (data, target) in enumerate(self.train_loader):
                iteration += 1
                data = data.to(self.device)
                target = target.to(self.device)
                self.optimizer.zero_grad()
                output = self.student(data)
                # Standard Learning Loss ( Classification Loss)
                # print("output:", output.size())
                # print("target:", target.size())
                loss_SL = criterion(output, target)

                loss = loss_SL
                # teacher_outputs = self.teacher(data)
                # fitnet_loss = fitnet_criterion(teacher_outputs, output)
                #
                # loss = fitnet_loss
                # loss.backward()
                # self.optimizer.step()

                if self.have_teacher:
                    teacher_outputs = self.teacher(data)


                    loss_KD = nn.KLDivLoss()(F.log_softmax(output / T, dim=1),
                                                      F.softmax(teacher_outputs / T, dim=1))
                    loss = (1 - lambda_) * loss_SL + lambda_ * T * 2 * T * loss_KD

                print("Batch %d of %d , loss %.3f"%(batch_idx, total_batches, loss),end="\r")
                #print('before loss backward')
                #total_loss.backward()
                loss.backward(retain_graph=True)
                self.optimizer.step()
              ###  在列表中记录当前的训练信息
                # train_logs.append({
                #     'epoch': epoch + 1,
                #     'batch_idx': batch_idx + 1,
                #     'total_batches': total_batches,
                #     'loss': loss.item()
                # })

            print("epoch {}/{}".format(epoch, epochs))
            val_acc = self.validate(step=epoch)
            if val_acc > best_acc:
                best_acc = val_acc
                self.save(epoch, name='states/{}_{}_best.pth.tar'.format(self.name, trial_id))

            self.accuracy_history.append(val_acc)  ## json
            self.save_accuracy()  ## json
            save_path = '/mnt/work2/soark2024/tabnn/my_json/'

            # setup_logging(os.path.join(save_path, 'logger.log'))
            # logging.info("saving to %s", save_path)
            # logging,debug("run arguments: %s", args)
           ###
        # df = pd.DataFrame(train_logs)
        # df.to_csv('/mnt/quantiKD/TA-bnn/states/indv/train_logs.csv', index=False)
        return best_acc

    def save_accuracy(self):
        accuracy_file = f'accuracy_{self.name}_{self.config["trial_id"]}.json'
        with open(accuracy_file, 'w') as f:
            json.dump(self.accuracy_history, f)


    def validate(self, step=0):
        self.student.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            acc = 0
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.student(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            # self.accuracy_history.append(acc)
            acc = 100 * correct / total

            print('{{"metric": "{}_val_accuracy", "value": {}}}'.format(self.name, acc))
            return acc

    def save(self, epoch, name=None):
        trial_id = self.config['trial_id']
        if name is None:
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.student.module.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, 'states/{}_{}_epoch{}.pth.tar'.format(self.name, trial_id, epoch))
        else:
            torch.save({
                'model_state_dict': self.student.module.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epoch': epoch,
            }, name)




    def adjust_learning_rate(self, optimizer, epoch):
        epochs = self.config['epochs']
# epoch for cifar - 80,120,180(240); for ImageNet 12, 28, 42 (60)
        if epoch < 30:
            lr = 0.01
        elif epoch < 60:
            lr = 0.01 * 0.1
        elif epoch < 90:
            lr = 0.01 * 0.01
        # elif epoch < 240:
        #     lr = 0.1 * 0.001
        else:
            lr = 0.01 * 0.001

        # if epoch < int(epoch/2.0):
        #     lr = 0.1
        # elif epoch < int(epochs*3/4.0):
        #     lr = 0.1 * 0.001
        # else:
        #     lr = 0.1 * 0.0001

        # update_list = [80, 130, 180, 220]
        # if epoch in update_list:
        #      for param_group in optimizer.param_groups:
        #          param_group['lr'] = param_group["lr"] * 0.1

      #  update optimizer's learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


# Train Teacher if provided a teacher, otherwise it's a normal training using only cross entropy loss
# This is for training single models(NOKD in paper) for baselines models (or training the first teacher)
def train_teacher(args, train_config):
    dataset = train_config['dataset']
    teacher_model = get_quant_model(args.teacher, (args.teacher_wbits, args.teacher_abits, args.teacher_quantization), dataset, use_cuda=args.cuda)
    if args.teacher_checkpoint:
        print("---------- Loading Teacher -------")
        teacher_model = load_checkpoint(teacher_model, args.teacher_checkpoint)
        # teacher_model = train_teacher(args, train_config)
        # teacher_layer = teacher_model.layer_to_extract
    else:
        print("---------- Training Teacher -------")
        train_loader, test_loader = get_dataset(dataset)
        teacher_train_config = copy.deepcopy(train_config)
        teacher_name = 'states/{}_{}_best.pth.tar'.format(args.teacher, train_config['trial_id'])
        teacher_train_config['name'] = args.teacher
        teacher_trainer = TrainManager(teacher_model, teacher=None, train_loader=train_loader, test_loader=test_loader, train_config=teacher_train_config)
        teacher_trainer.train()
        teacher_model = load_checkpoint(teacher_model, os.path.join('./', teacher_name))
    return teacher_model

def train_student(args, train_config, teacher_model=None):
    dataset = train_config['dataset']
    student_model = get_quant_model(args.student, (args.student_wbits, args.student_abits, args.student_quantization), dataset, use_cuda=args.cuda)
    # Student training
    if teacher_model == None:
        print("---------- No Teacher -------------")
        print("---------- Training Student -------")
    else:
        params1 = teacher_model.named_parameters()
        params2 = student_model.named_parameters()
        dict_params2 = dict(params2)
        for name1, param1 in params1:
            if name1 in dict_params2:
                dict_params2[name1].data.copy_(param1.data)
        print("---------- Training Student -------")
    student_train_config = copy.deepcopy(train_config)
    train_loader, test_loader = get_dataset(dataset)
    student_train_config['name'] = args.student
    student_trainer = TrainManager(student_model, teacher=teacher_model, train_loader=train_loader, test_loader=test_loader, train_config=student_train_config)
    best_student_acc = student_trainer.train()
    print(f'best_student_acc: {best_student_acc}')
    return student_model


