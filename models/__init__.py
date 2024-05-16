from .simclr_model import SimCLR
from .clip_model import CLIP
from .imagenet_model import ImageNetResNet
from lightly.models import ResNetGenerator
import torch.nn as nn


def get_encoder_architecture(args):
    if args.pretraining_dataset == 'cifar10':
        return SimCLR()
    elif args.pretraining_dataset == 'stl10':
        return SimCLR()
    elif args.pretraining_dataset == 'gtsrb':
        return SimCLR()
    elif args.pretraining_dataset == 'svhn':
        return SimCLR()
    elif args.pretraining_dataset == 'tiny_imagenet':
        return SimCLR()
    else:
        raise ValueError('Unknown pretraining dataset: {}'.format(args.pretraining_dataset))


def get_encoder_architecture_usage(args):
    if args.encoder_usage_info == 'cifar10':
        return SimCLR()
    elif args.encoder_usage_info == 'stl10':
        return SimCLR()
    elif args.encoder_usage_info == 'gtsrb':
        return SimCLR()
    elif args.encoder_usage_info == 'svhn':
        return SimCLR()
    elif args.encoder_usage_info == 'imagenet':
        return ImageNetResNet()
    elif args.encoder_usage_info == 'CLIP':
        return CLIP(1024, 224, vision_layers=(3, 4, 6, 3), vision_width=64)
    elif args.encoder_usage_info in ['MOCO']:
        resnet = ResNetGenerator("resnet-18", 1, num_splits=8)
        backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.AdaptiveAvgPool2d(1),
        )
        return backbone
    elif args.encoder_usage_info in ['simsiam','swav','byol','NNCLR','DINO']:
        resnet = ResNetGenerator("resnet-18")
        backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.AdaptiveAvgPool2d(1),
        )
        return backbone
    else:
        raise ValueError('Unknown pretraining dataset: {}'.format(args.pretraining_dataset))
