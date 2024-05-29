## The paper: Optimizing Binary Post-Training Quantization with Assistive Teaching and Coarse-grained Pruning. (Submitted to IEEE SPL)

### Training teacher or simple student with the command:
python main.py --epochs 160 --student resnet20 --student-wbits 32 --student-abits 32 --dataset imagenet --trial-id 'imagenet-32bit'
python main.py --epochs 160 --student resnet20 --student-wbits 1 --student-abits 1 --dataset imagenet --trial-id 'imagenet-1bit'

### Training ASBQ through 32bit -> 8bit -> 4bit -> 2bit -> 1bit:
python main.py --epochs 100 --teacher resnet18 --teacher-checkpoint /mnt/quantiKD/TA-bnn/states/your_checkpoint_position_in_here_best.pth.tar --teacher-wbits 32 --teacher-abits 32 --student resnet18 --student-wbits 8 --student-abits 8 --dataset imagenet --trial-id 'imagenet-32-8bit' --weight-decay 1e-4

### From our experience with hyperparameters in CIFAR-10/100:
* epoch: 240;
* Batch size: 128;
* LR: 32bit,8bit,4bit with {80,120,180} using {0.1, 0.1x0.01, 0.1x0.001}; 2bit,1bit with {80,120,180} using {0.01, 0.01x0.01, 0.01x0.001}.

### From our experience with hyperparameters in ImageNet:
* epoch: 32bit,1bit is 100 epoch, others 60 epoch;
* Batch size: 256;
* LR: 32bit with {30,60,90} using {0.1, 0.1x0.1, 0.1x0.01}.

### Final the 1bit student results in the ImageNet:
<img width="340" alt="result12" src="https://github.com/Luadoo/ASBQ/assets/58927660/63fcbaa7-ca41-4943-a1e0-9118f6e11cd2"> 
<img width="400" alt="result123" src="https://github.com/Luadoo/ASBQ/assets/58927660/4b8679ab-5b26-44e2-b768-700583128d39">

### Reference:
* https://github.com/666DZY666/micronet
* B. Zhuang, M. Tan, J. Liu, L. Liu, I. Reid, and C. Shen, “Effective training of convolutional neural networks with low-bitwidth weights and activations,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 44, no. 10, pp. 6140–6152, 2021.  (DOI: 10.1109/TPAMI.2021.3088904)
* S. Han, H. Mao, and W. J. Dally, “Deep compression: Compressing deep neural networks with pruning, trained quantization and huffman coding,” arXiv preprint arXiv:1510.00149, 2015.
* H. Wu, R. He, H. Tan, X. Qi, and K. Huang, “Vertical layering of quantized neural networks for heterogeneous inference,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2023. (DOI:10.1109/TPAMI.2023.3319045)
