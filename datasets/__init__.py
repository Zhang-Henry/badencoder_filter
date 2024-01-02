import torch
import torchvision

from .cifar10_dataset import get_pretraining_cifar10, get_shadow_cifar10, get_downstream_cifar10, get_shadow_cifar10_224
from .gtsrb_dataset import  get_downstream_gtsrb
from .svhn_dataset import get_downstream_svhn
from .stl10_dataset import get_pretraining_stl10, get_shadow_stl10, get_downstream_stl10
from .imagenet_dataset_ import get_shadow_imagenet
from .tiny_imagenet_datatset import get_pretraining_imagenet

def get_pretraining_dataset(args):
    if args.pretraining_dataset == 'cifar10':
        return get_pretraining_cifar10(args.data_dir)
    elif args.pretraining_dataset == 'stl10':
        return get_pretraining_stl10(args.data_dir)
    elif args.pretraining_dataset == 'tiny_imagenet':
        return get_pretraining_imagenet(args.data_dir)
    else:
        raise NotImplementedError


def get_shadow_dataset(args):
    if args.shadow_dataset =='cifar10':
        return get_shadow_cifar10(args)
    elif args.shadow_dataset == 'stl10':
        return get_shadow_stl10(args)
    elif args.shadow_dataset == 'imagenet':
        return get_shadow_imagenet(args)
    else:
        raise NotImplementedError


def get_dataset_evaluation(args):
    if args.dataset =='cifar10':
        return get_downstream_cifar10(args)
    elif args.dataset == 'gtsrb':
        return get_downstream_gtsrb(args)
    elif args.dataset == 'svhn':
        return get_downstream_svhn(args)
    elif args.dataset == 'stl10':
        return get_downstream_stl10(args)
    else:
        raise NotImplementedError
