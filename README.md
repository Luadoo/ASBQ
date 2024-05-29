### The paper: Optimizing Binary Post-Training Quantization with Assistive Teaching and Coarse-grained Pruning. (Subbitted to IEEE SPL)

# Training teacher or simple student with command:
python main.py --epochs 160 --student resnet20 --student-wbits 32 --student-abits 32 --dataset imagenet --trial-id 'imagenet-32bit'
python main.py --epochs 160 --student resnet20 --student-wbits 1 --student-abits 1 --dataset imagenet --trial-id 'imagenet-1bit'

# Training ASBQ throught 32bit -> 8bit -> 4bit -> 2bit -> 1bit:
python main.py --epochs 100 --teacher resnet18 --teacher-checkpoint /mnt/quantiKD/TA-bnn/states/you_checkpoint_position_in_here_best.pth.tar --teacher-wbits 32 --teacher-abits 32 --student resnet18 --student-wbits 8 --student-abits 8 --dataset imagenet --trial-id 'imagenet-32-8bit' --weight-decay 1e-4
