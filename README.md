## The paper: Optimizing Binary Post-Training Quantization with Assistive Teaching and Coarse-grained Pruning. (Submitted to IEEE SPL)

### Training teacher or simple student with the command:
python main.py --epochs 160 --student resnet20 --student-wbits 32 --student-abits 32 --dataset imagenet --trial-id 'imagenet-32bit'
python main.py --epochs 160 --student resnet20 --student-wbits 1 --student-abits 1 --dataset imagenet --trial-id 'imagenet-1bit'

### Training ASBQ throught 32bit -> 8bit -> 4bit -> 2bit -> 1bit:
python main.py --epochs 100 --teacher resnet18 --teacher-checkpoint /mnt/quantiKD/TA-bnn/states/you_checkpoint_position_in_here_best.pth.tar --teacher-wbits 32 --teacher-abits 32 --student resnet18 --student-wbits 8 --student-abits 8 --dataset imagenet --trial-id 'imagenet-32-8bit' --weight-decay 1e-4

### From our experience with hyperparameters in CIFAR-10/100

epoch: 240
Batch size: 128
LR: 32bit,8bit,4bit with {80,120,180} using {0.1, 0.1*0.01, 0.1*0.001}; 2bit,1bit with {80,120,180} using {0.01, 0.01*0.01, 0.01*0.001}

### From our experience with hyperparameters in ImageNet:
epoch: 32bit is 100 epoch, others 60 epoch when distillation to 1bit is 100 epoch.
Batch size: 256
LR: 32bit with {30,60,90} using {0.1, 0.1*0.1, 0.1*0.01}


