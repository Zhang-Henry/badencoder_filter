import sys
sys.path.append("..")

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from datasets.backdoor_dataset import CIFAR10M
import numpy as np
from datasets.bd_dataset_imagenet_filter import BadEncoderDataset


def cifar10_dataloader(args):
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ])

    clean_transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
                ])

    memory_data = CIFAR10M(numpy_file='../data/cifar10/train.npz', class_type=classes, transform=train_transform,transform2=clean_transform)
    train_loader = DataLoader(memory_data, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    return train_loader


def stl10_dataloader(args):
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.44087798, 0.42790666, 0.38678814], [0.25507198, 0.24801506, 0.25641308])])

    clean_transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.44087798, 0.42790666, 0.38678814], [0.25507198, 0.24801506, 0.25641308])])


    memory_data = CIFAR10M(numpy_file='../data/stl10/train.npz', class_type=classes, transform=train_transform,transform2=clean_transform)
    train_loader = DataLoader(memory_data, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    return train_loader


def imagenet_dataloader(args):

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(256),
        transforms.CenterCrop(size=(224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor()
        ])

    clean_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    classes = [str(i) for i in range(1000)]


    training_data_num = 1000000
    # np.random.seed(100)
    training_data_sampling_indices = np.random.choice(training_data_num, int(training_data_num*0.01), replace=False)

    shadow_dataset = BadEncoderDataset(
        root = "../data/imagenet/train",
        class_type=classes,indices = training_data_sampling_indices,
        transform=clean_transform,
        bd_transform=train_transform,
    )
    train_loader = DataLoader(shadow_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    return train_loader
