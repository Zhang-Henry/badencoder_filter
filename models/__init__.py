from .simclr_model import SimCLR
from .clip_model import CLIP
from .imagenet_model import ImageNetResNet
from lightly.models import ResNetGenerator
import torch.nn as nn
from timm.models.vision_transformer import vit_base_patch16_224
from lightly.models.modules import MaskedVisionTransformerTIMM
from torchvision.models import resnet50,resnet18


def get_encoder_architecture(args):
    if args.pretraining_dataset == 'cifar10':
        return SimCLR()
    elif args.pretraining_dataset == 'stl10':
        return SimCLR()
    else:
        raise ValueError('Unknown pretraining dataset: {}'.format(args.pretraining_dataset))


def get_encoder_architecture_usage(args):
    if args.encoder_usage_info in ['cifar10','stl10','gtsrb','svhn']:
        print('Using SimCLR for CIFAR10, STL10, GTSRB, SVHN')
        return SimCLR()
    elif args.encoder_usage_info == 'imagenet':
        print('Using ImageNetResNet for ImageNet')
        return ImageNetResNet()
    elif args.encoder_usage_info == 'CLIP':
        return CLIP(1024, 224, vision_layers=(3, 4, 6, 3), vision_width=64)
    elif args.encoder_usage_info in ['MOCO']:
        resnet = ResNetGenerator("resnet-18", 1, num_splits=8)
        backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.AdaptiveAvgPool2d(1),
        )
        print('Using resnet18 for MoCo')
    elif args.encoder_usage_info in ['simsiam','swav','byol','NNCLR','DINO']:
        resnet = ResNetGenerator("resnet-18")
        backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.AdaptiveAvgPool2d(1),
        )
        print('Using resnet18 for simsiam,swav,byol,NNCLR,DINO')
    elif args.encoder_usage_info in ['mae']:
        vit = vit_base_patch16_224()
        backbone = MaskedVisionTransformerTIMM(vit=vit)
        print('Using ViT for MAE')
    elif args.encoder_usage_info in ['mocov2']:
        resnet = resnet50()
        resnet.fc = nn.Identity()  # Ignore classification head
        backbone = resnet
        print('Using resnet50 for MoCoV2')
    elif args.encoder_usage_info in ['imagenet_100']:
        resnet = resnet18()
        backbone = nn.Sequential(*list(resnet.children())[:-1])
        print('Using resnet18 for imagenet_100')
    else:
        raise ValueError('Unknown pretraining dataset: {}'.format(args.pretraining_dataset))

    shadow_dataset = getattr(args, 'shadow_dataset', None)
    dataset = getattr(args, 'dataset', None)
    print('shadow_dataset:',shadow_dataset)
    print('dataset:',dataset)
    ##### imagenet_100作为pretrain数据集，byol,simsiam做encoder，也使用resnet18作为backbone，但这个是torchvision里的resnet18
    if (args.encoder_usage_info in ['byol','simsiam'] and shadow_dataset=='imagenet_100') \
        or (args.encoder_usage_info in ['byol','simsiam'] and dataset=='imagenet_100'):
        resnet = resnet18()
        backbone = nn.Sequential(*list(resnet.children())[:-1])
        print('Using resnet18 for imagenet_100')

    return backbone