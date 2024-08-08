from torchvision import datasets, transforms
from .backdoor_dataset_imagenet import BadEncoderDataset,ImageNetMem,BadEncoderImageNetBackdoor
import numpy as np
from lightly.transforms import utils
from lightly.transforms import MAETransform
from torch.utils.data import Dataset
from lightly.transforms.moco_transform import MoCoV2Transform

from .backdoor_dataset import ReferenceImg
from .trans import *
from lightly.transforms import (
    BYOLTransform,
    DINOTransform,
    FastSiamTransform,
    SimCLRTransform,
    SimSiamTransform,
    SwaVTransform,
)

transform_imagenet = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(utils.IMAGENET_NORMALIZE["mean"], utils.IMAGENET_NORMALIZE["std"])
    ])



test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(utils.IMAGENET_NORMALIZE["mean"], utils.IMAGENET_NORMALIZE["std"])
])

# transform_imagenet = transforms.Compose([
#     transforms.Resize(size=256),
#     transforms.CenterCrop(size=(224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.4850, 0.4560, 0.4060], [0.2290, 0.2240, 0.2250])])


# finetune_transform = transforms.Compose([
#     transforms.RandomHorizontalFlip(p=0.5),
#     transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
#     transforms.RandomGrayscale(p=0.2),
#     transforms.ToTensor(),
#     # transforms.Normalize([0.4850, 0.4560, 0.4060], [0.2290, 0.2240, 0.2250])
#     ])

#classes = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']
classes = [str(i) for i in range(1000)]






# def get_shadow_imagenet(args):
#     training_data_num = 1000000
#     # np.random.seed(100)
#     #training_data_sampling_indices = np.random.choice(training_data_num, training_data_num, replace=False)
#     training_data_sampling_indices = np.random.choice(training_data_num, int(training_data_num*0.01), replace=False)

#     shadow_dataset = BadEncoderDataset(
#         root = "data/imagenet/train",
#         trigger_file=args.trigger_file,
#         reference_file= args.reference_file,
#         class_type=classes,indices = training_data_sampling_indices,
#         transform=train_transform,
#         bd_transform=backdoor_transform,
#         ftt_transform=finetune_transform
#     )

#     return shadow_dataset, None, None, None


def get_shadow_imagenet(args):
    training_data_num = 100000
    # np.random.seed(100)
    #training_data_sampling_indices = np.random.choice(training_data_num, training_data_num, replace=False)
    training_data_sampling_indices = np.random.choice(training_data_num, int(training_data_num*0.01), replace=False)

    if args.encoder_usage_info == 'imagenet':
        train_transform = train_transform
    elif args.encoder_usage_info == 'mae':
        train_transform = MAETransform()
    elif args.encoder_usage_info == 'mocov2':
        train_transform = MoCoV2Transform(input_size=224)

    shadow_dataset = BadEncoderDataset(
        root = "data/imagenet/train",
        trigger_file=args.trigger_file,
        reference_file= args.reference_file,
        class_type=classes,indices = training_data_sampling_indices,
        transform=train_transform,
        bd_transform=transform_imagenet,
        ftt_transform=test_transform
    )

    return shadow_dataset, None, None, None





def get_downstream_imagenet(args):
    imagenet_train_dataset = datasets.ImageNet(root='data/imagenet',split='train', transform=test_transform)
    imagenet_val_dataset = datasets.ImageNet(root='data/imagenet',split='val', transform=test_transform)

    if args.encoder_usage_info == 'mocov2':
        print('test_transform_imagenet')
        memory_data = ImageNetMem(imagenet_train_dataset, class_type=classes)
        test_data_backdoor = BadEncoderImageNetBackdoor(imagenet_val_dataset, reference_label=args.reference_label)
        test_data_clean = ImageNetMem(imagenet_val_dataset, class_type=classes)

    else:
        raise NotImplementedError

    target_dataset = ReferenceImg(reference_file=args.reference_file, transform=test_transform)


    return target_dataset, memory_data, test_data_clean, test_data_backdoor



def get_shadow_imagenet_100(args):
    training_data_num = 100000
    # np.random.seed(100)
    #training_data_sampling_indices = np.random.choice(training_data_num, training_data_num, replace=False)
    training_data_sampling_indices = np.random.choice(training_data_num, int(training_data_num*0.01), replace=False)

    if args.encoder_usage_info == 'imagenet_100':
        train_transform = SimCLRTransform()
    elif args.encoder_usage_info == 'simsiam':
        train_transform = SimSiamTransform()
    elif args.encoder_usage_info == 'byol':
        train_transform = BYOLTransform()

    shadow_dataset = BadEncoderDataset(
        root = "data/imagenet_100/train",
        trigger_file=args.trigger_file,
        reference_file= args.reference_file,
        class_type=classes,indices = training_data_sampling_indices,
        transform=train_transform,
        bd_transform=transform_imagenet,
        ftt_transform=test_transform
    )

    return shadow_dataset, None, None, None





def get_downstream_imagenet_100(args):
    imagenet_train_dataset = datasets.ImageNet(root='data/imagenet_100',split='train', transform=test_transform)
    imagenet_val_dataset = datasets.ImageNet(root='data/imagenet_100',split='val', transform=test_transform)

    if args.encoder_usage_info in ['mocov2','imagenet_100','simsiam','byol']:
        print('test_transform_imagenet')
        memory_data = ImageNetMem(imagenet_train_dataset, class_type=classes)
        test_data_backdoor = BadEncoderImageNetBackdoor(imagenet_val_dataset, reference_label=args.reference_label)
        test_data_clean = ImageNetMem(imagenet_val_dataset, class_type=classes)

    else:
        raise NotImplementedError

    target_dataset = ReferenceImg(reference_file=args.reference_file, transform=test_transform)


    return target_dataset, memory_data, test_data_clean, test_data_backdoor
