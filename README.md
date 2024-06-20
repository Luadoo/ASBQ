## The paper: Optimizing Binary Post-Training Quantization with Assistive Teaching and Coarse-grained Pruning. (Submitted to IEEE SPL)

### Training teacher or simple student with the command:
* python main.py --epochs 240 --student resnet20 --student-wbits 32 --student-abits 32 --dataset imagenet --trial-id 'imagenet-32bit'
* python main.py --epochs 240 --student resnet20 --student-wbits 1 --student-abits 1 --dataset imagenet --trial-id 'imagenet-1bit'

### Training ASBQ through 32bit -> 8bit -> 4bit -> 2bit -> 1bit:
python main.py --epochs 100 --teacher resnet18 --teacher-checkpoint /mnt/quantiKD/TA-bnn/states/your_checkpoint_position_in_here_best.pth.tar --teacher-wbits 32 --teacher-abits 32 --student resnet18 --student-wbits 8 --student-abits 8 --dataset imagenet --trial-id 'imagenet-32-8bit' --weight-decay 1e-4

### From our experience with hyperparameters in CIFAR-10/100:
* epoch: 240;
* Batch size: 128;
* LR: 32bit,8bit,4bit with {80,120,180} using {0.1, 0.1x0.01, 0.1x0.001}; 2bit,1bit with {80,120,180} using {0.01, 0.01x0.01, 0.01x0.001}.

### From our experience with hyperparameters in ImageNet:
* epoch: 32bit,1bit is 100 epoch, others 60 epoch;
* Batch size: 256;
* LR: 32bit with {30,60,90} using {0.1, 0.1x0.1, 0.1x0.01}; 1bit with {30,60,90} using {0.01, 0.01x0.1, 0.01x0.01}.

#### The ResNet-20 checkpoint. pth and JSON.file results:
* [resnet20-cifar10.zip](https://github.com/user-attachments/files/15507611/resnet20-cifar10.zip) (each training stage was take 2hours finished.)
* [accuracy_resnet20_new-2-1bit-new-0.1.json](https://github.com/user-attachments/files/15507553/accuracy_resnet20_new-2-1bit-new-0.1.json)[81.99, 83.71, 84.25, 84.95, 85.05, 85.43, 85.18, 86.04, 85.68, 86.13, 86.02, 85.82, 86.01, 86.01, 86.03, 86.06, 86.55, 86.53, 86.68, 86.41, 86.63, 86.78, 86.9, 86.79, 86.68, 87.01, 87.14, 86.47, 86.72, 86.53, 86.62, 86.83, 86.92, 86.64, 86.56, 87.01, 87.31, 86.29, 87.05, 86.69, 86.46, 86.87, 86.12, 86.89, 87.6, 86.54, 86.76, 86.94, 86.77, 86.78, 87.11, 87.14, 87.13, 86.76, 86.7, 86.58, 86.93, 86.85, 86.91, 86.86, 86.92, 86.41, 86.45, 87.02, 87.06, 87.19, 87.61, 87.14, 86.81, 87.01, 87.15, 86.97, 86.56, 86.63, 86.91, 86.84, 86.82, 87.12, 86.96, 86.73, 87.55, 88.11, 88.13, 87.97, 87.69, 87.99, 87.65, 87.51, 87.89, 87.91, 87.67, 88.06, 87.77, 87.79, 87.48, 88.47, 87.88, 87.93, 87.43, 88.05, 87.75, 87.61, 87.89, 87.64, 87.8, 87.24, 87.93, 87.65, 87.37, 88.3, 87.58, 87.62, 87.87, 87.96, 87.84, 87.48, 87.87, 87.82, 88.14, 87.93, 87.64, 88.07, 87.93, 87.81, 87.37, 88.23, 87.74, 87.84, 87.63, 88.12, 87.97, 87.99, 88.37, 87.89, 87.72, 87.77, 88.02, 87.78, 87.83, 88.0, 87.93, 87.45, 88.07, 87.86, 87.99, 88.09, 87.75, 88.12, 87.65, 87.87, 87.97, 87.9, 87.92, 87.81, 88.02, 87.81, 87.99, 87.94, 87.82, 88.33, 87.71, 88.09, 88.27, 87.75, 87.58, 87.88, 87.81, 87.78, 87.61, 87.96, 87.8, 87.76, 88.04, 88.0, 87.75, 87.69, 87.91, 87.95, 88.2, 87.83, 87.87, 87.71, 87.92, 88.09, 87.7, 87.88, 87.99, 87.56, 87.86, 87.97, 88.0, 87.97, 87.68, 87.85, 88.03, 88.06, 87.99, 88.38, 88.15, 87.88, 87.75, 88.11, 87.72, 87.79, 88.19, 87.69, 87.58, 87.86, 87.93, 87.83, 87.67, 88.54, 88.12, 87.67, 88.09, 88.15, 87.83, 87.8, 88.08, 87.78, 87.93, 88.04, 88.1, 87.9, 88.26, 87.43, 88.19, 87.61, 87.86, 88.13, 87.88, 87.85, 88.11, 87.79, 88.09, 87.86, 88.0, 87.81, 87.98, 88.07] 

### Final the 1bit student results in the ImageNet:
* Our ASBQ outperforms recent work LKBQ (PTQ) methods, we push the limit of QAT in BNN using knowledge transfer with 10.37% outperforms.
<img width="388" alt="github1" src="https://github.com/Luadoo/ASBQ/assets/58927660/19736dd7-2221-48f7-96c5-99f260359aa0">
<img width="388" alt="github1" src="https://github.com/Luadoo/ASBQ/assets/58927660/44e2a42a-8b13-4b32-b2f1-2be27ce95209">





### Reference:
* https://github.com/666DZY666/micronet
* B. Zhuang, M. Tan, J. Liu, L. Liu, I. Reid, and C. Shen, “Effective training of convolutional neural networks with low-bitwidth weights and activations,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 44, no. 10, pp. 6140–6152, 2021.  (DOI: 10.1109/TPAMI.2021.3088904)
* S. Han, H. Mao, and W. J. Dally, “Deep compression: Compressing deep neural networks with pruning, trained quantization and huffman coding,” arXiv preprint arXiv:1510.00149, 2015.
* H. Wu, R. He, H. Tan, X. Qi, and K. Huang, “Vertical layering of quantized neural networks for heterogeneous inference,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2023. (DOI:10.1109/TPAMI.2023.3319045)

### Contact:
Email: soark@ajou.ac.kr

