from torchvision import transforms
from .backdoor_dataset import *
import numpy as np
import torch
from .trans import *
import torch.nn.functional as F
from lightly.transforms import (
    BYOLTransform,
    BYOLView1Transform,
    BYOLView2Transform,
    DINOTransform,
    FastSiamTransform,
    SimCLRTransform,
    SimSiamTransform,
    SMoGTransform,
    SwaVTransform,
    MoCoV2Transform
)

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])


finetune_transform_cifar10 = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

finetune_transform_CLIP = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),])

backdoor_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])


classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def get_pretraining_cifar10(data_dir):

    train_data = CIFAR10Pair(numpy_file=data_dir + "train.npz", class_type= classes, transform=train_transform)
    memory_data = CIFAR10Mem(numpy_file=data_dir + "train.npz", class_type= classes, transform=test_transform_cifar10)
    test_data  = CIFAR10Mem(numpy_file=data_dir + "test.npz", class_type= classes,transform=test_transform_cifar10)

    return train_data, memory_data, test_data


def get_shadow_cifar10(args):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ])
    if args.encoder_usage_info == 'cifar10':
        train_transform = train_transform
    elif args.encoder_usage_info == 'MOCO':
        train_transform = MoCoV2Transform(
            input_size=32,
            gaussian_blur=0.0,
        )
    elif args.encoder_usage_info == 'simsiam':
        train_transform = SimSiamTransform(
            input_size=32,
            gaussian_blur=0.0,
        )
    elif args.encoder_usage_info == 'swav':
        train_transform = SwaVTransform(
            crop_sizes=[32],
            crop_max_scales=[32],
            crop_counts=[2],  # 2 crops @ 32x32px
            crop_min_scales=[0.14],
            cj_strength=0.5,
            gaussian_blur=0,
        )
    elif args.encoder_usage_info == 'byol':
        train_transform = BYOLTransform(
            view_1_transform=BYOLView1Transform(input_size=32, gaussian_blur=0.0),
            view_2_transform=BYOLView2Transform(input_size=32, gaussian_blur=0.0),
        )
    elif args.encoder_usage_info == 'NNCLR':
        train_transform = SimCLRTransform(
            input_size=32,
            cj_strength=0.5,
            gaussian_blur=0.0,
        )
    elif args.encoder_usage_info == 'DINO':
        train_transform = DINOTransform(
            global_crop_size=32,
            n_local_views=0,
            cj_strength=0.5,
            gaussian_blur=(0, 0, 0),
        )

    training_data_num = 50000
    testing_data_num = 10000
    np.random.seed(100)
    #print('number of training examples:')
    training_data_sampling_indices = np.random.choice(training_data_num, training_data_num, replace=False)
    print('loading from the training data')
    shadow_dataset = BadEncoderDataset(
        numpy_file=args.data_dir + 'train.npz',
        trigger_file=args.trigger_file,
        reference_file= args.reference_file,
        class_type=classes,
        indices = training_data_sampling_indices,
        transform=train_transform,
        bd_transform=test_transform_cifar10,
        ftt_transform=finetune_transform_cifar10
    )
    memory_data = CIFAR10Mem(numpy_file=args.data_dir+'train.npz', class_type=classes, transform=test_transform_cifar10)
    test_data_backdoor = BadEncoderTestBackdoor(numpy_file=args.data_dir+'test.npz', trigger_file=args.trigger_file, reference_label= args.reference_label,  transform=test_transform_cifar10)
    # test_data_backdoor = BadEncoderTestBackdoor(numpy_file=args.data_dir+'test.npz', trigger_file=args.trigger_file, reference_label= args.reference_label,  transform=test_transform_cifar10_bd)
    test_data_clean = CIFAR10Mem(numpy_file=args.data_dir+'test.npz', class_type=classes, transform=test_transform_cifar10)

    return shadow_dataset, memory_data, test_data_clean, test_data_backdoor


def get_shadow_cifar10_224(args):
    training_data_num = 50000
    testing_data_num = 10000
    np.random.seed(100)
    training_data_sampling_indices = np.random.choice(training_data_num, training_data_num, replace=False)
    print('loading from the training data')

    shadow_dataset = BadEncoderDataset(
        numpy_file=args.data_dir+'train_224.npz',
        trigger_file=args.trigger_file,
        reference_file= args.reference_file,
        class_type=classes,
        indices = training_data_sampling_indices,
        transform=finetune_transform_CLIP,
        bd_transform=test_transform_CLIP,
        ftt_transform=finetune_transform_CLIP
    )

    return shadow_dataset, None, None, None


def get_downstream_cifar10(args):
    training_file_name = 'train.npz'
    testing_file_name = 'test.npz'

    if args.encoder_usage_info == 'cifar10':
        if args.noise == 'GaussianBlur':
            test_transform = test_transform_cifar10_GaussianBlur
            print('test_transform_cifar10_GaussianBlur')
        elif args.noise == 'JPEGcompression':
            test_transform = test_transform_cifar10_JPEGcompression
            print('test_transform_cifar10_JPEGcompression')
        elif args.noise == 'salt_and_pepper_noise':
            test_transform = test_transform_cifar10_salt_and_pepper_noise
            print('test_transform_cifar10_salt_and_pepper_noise')
        elif args.noise == 'poisson_noise':
            test_transform = test_transform_cifar10_poisson_noise
            print('test_transform_cifar10_poisson_noise')
        else:
            test_transform = test_transform_cifar10
            print('test_transform_cifar10')
        memory_data = CIFAR10Mem(numpy_file=args.data_dir+training_file_name, class_type=classes, transform=test_transform)
        test_data_backdoor = BadEncoderTestBackdoor(numpy_file=args.data_dir+testing_file_name, trigger_file=args.trigger_file, reference_label= args.reference_label,  transform=test_transform)
        test_data_clean = CIFAR10Mem(numpy_file=args.data_dir+testing_file_name, class_type=classes, transform=test_transform)
    elif args.encoder_usage_info == 'stl10':
        if args.noise == 'GaussianBlur':
            test_transform = test_transform_stl10_GaussianBlur
            print('test_transform_stl10_GaussianBlur')
        elif args.noise == 'JPEGcompression':
            test_transform = test_transform_stl10_JPEGcompression
            print('test_transform_stl10_JPEGcompression')
        elif args.noise == 'salt_and_pepper_noise':
            test_transform = test_transform_stl10_salt_and_pepper_noise
            print('test_transform_stl10_salt_and_pepper_noise')
        elif args.noise == 'poisson_noise':
            test_transform = test_transform_stl10_poisson_noise
            print('test_transform_stl10_poisson_noise')
        else:
            test_transform = test_transform_stl10
            print('test_transform_stl10')
        memory_data = CIFAR10Mem(numpy_file=args.data_dir+training_file_name, class_type=classes, transform=test_transform)
        test_data_backdoor = BadEncoderTestBackdoor(numpy_file=args.data_dir+testing_file_name, trigger_file=args.trigger_file, reference_label= args.reference_label,  transform=test_transform)
        test_data_clean = CIFAR10Mem(numpy_file=args.data_dir+testing_file_name, class_type=classes, transform=test_transform)
    elif args.encoder_usage_info == 'CLIP':
        print('test_transform_CLIP')
        test_transform = test_transform_CLIP
        training_file_name = 'train_224.npz'
        testing_file_name = 'test_224.npz'
        memory_data = CIFAR10Mem(numpy_file=args.data_dir+training_file_name, class_type=classes, transform=test_transform)
        test_data_backdoor = BadEncoderTestBackdoor(numpy_file=args.data_dir+testing_file_name, trigger_file=args.trigger_file, reference_label= args.reference_label,  transform=test_transform)
        test_data_clean = CIFAR10Mem(numpy_file=args.data_dir+testing_file_name, class_type=classes, transform=test_transform)
    elif args.encoder_usage_info == 'imagenet':
        print('test_transform_imagenet')
        test_transform = test_transform_imagenet
        training_file_name = 'train_224'
        testing_file_name = 'test_224'
        memory_data = CIFAR10Mem_224(numpy_file=args.data_dir+training_file_name, class_type=classes, transform=test_transform)
        test_data_backdoor = BadEncoderTestBackdoor_224(numpy_file=args.data_dir+testing_file_name, trigger_file=args.trigger_file, reference_label= args.reference_label,  transform=test_transform)
        test_data_clean = CIFAR10Mem_224(numpy_file=args.data_dir+testing_file_name, class_type=classes, transform=test_transform)
    else:
        raise NotImplementedError

    target_dataset = ReferenceImg(reference_file=args.reference_file, transform=test_transform)

    return target_dataset, memory_data, test_data_clean, test_data_backdoor
