from torchvision.datasets import CIFAR100
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder


def get_cifar100():
    pin_memory = True

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
      #  transforms.GaussianBlur(kernel_size=1, sigma=(0.1, 2.0)),
        # transforms.Pad(padding=4, padding_mode='reflect'),
     #   transforms.RandomResizedCrop(32),
        transforms.Normalize(mean=(0.5071, 0.4865, 0.4409), std=(0.2673, 0.2564, 0.2762)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5071, 0.4865, 0.4409), std=(0.2673, 0.2564, 0.2762)),
    ])

    trainset = CIFAR100(root='./data', train=True, download=True, transform=transform_train)

    train_loader = DataLoader(
        trainset, batch_size=128, shuffle=True,
        num_workers=4, pin_memory=pin_memory
    )

    testset = CIFAR100(root='./data', train=False, download=False, transform=transform_test)
    test_loader = DataLoader(
        testset, batch_size=128, shuffle=False,
        num_workers=4, pin_memory=pin_memory)
    
    return train_loader, test_loader

def get_dataset(dataset):
    if dataset == 'cifar100':
        return get_cifar100()
    else:
        raise Exception('only dataset is cifar100 currently!')


def get_cifar10():
    pin_memory = True

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
      #  transforms.RandomHorizontalFlip(0.5),           ##
        # transforms.RandomRotation(20),                ##
        transforms.ToTensor(),
    #    transforms.GaussianBlur(kernel_size=1, sigma=(0.1, 2.0)),
        # transforms.Pad(padding=4, padding_mode='reflect'),   ##
      #  transforms.RandomResizedCrop(32),                  ##
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    trainset = CIFAR10(root='./data', train=True, download=True, transform=transform_train)

    train_loader = DataLoader(
        trainset, batch_size=128, shuffle=True,
        num_workers=4, pin_memory=pin_memory
    )

    testset = CIFAR10(root='./data', train=False, download=False, transform=transform_test)
    test_loader = DataLoader(
        testset, batch_size=128, shuffle=False,
        num_workers=4, pin_memory=pin_memory)

    return train_loader, test_loader




def get_imagenet():
    pin_memory = True

    transform_train = transforms.Compose([
      #  transforms.Resize(224),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    trainset = ImageFolder(root='/mnt/work2/ImageNet/data/train', transform=transform_train)
    train_loader = DataLoader(
        trainset, batch_size=512, shuffle=True,
        num_workers=16, pin_memory=True
    )

    testset = ImageFolder(root='/mnt/work2/ImageNet/data/val', transform=transform_test)
    test_loader = DataLoader(
        testset, batch_size=512, shuffle=False,
        num_workers=16, pin_memory=True
    )

    return train_loader, test_loader


def get_dataset(dataset):
    if dataset == 'cifar10':
        return get_cifar10()
    elif dataset == 'cifar100':
        return get_cifar100()
    elif dataset == 'imagenet':
        return get_imagenet()
    else:
        raise Exception('dataset unknown!!')

train_loader, test_loader = get_imagenet()