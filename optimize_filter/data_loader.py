from torchvision import transforms
import moco.loader as loader
import moco.dataset as dataset
import torch


aug = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
    transforms.RandomApply([
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
    ], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([loader.GaussianBlur([.1, 2.])], p=0.5),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])
])

# aug = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.RandomApply([
#         transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
#     ], p=0.8),
#     transforms.RandomGrayscale(p=0.2),
#     transforms.RandomApply([loader.GaussianBlur([.1, 2.])], p=0.5),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     # transforms.Normalize(mean=[0.485, 0.456, 0.406],
#     #                                  std=[0.229, 0.224, 0.225])
# ])



def create_data_loader(args):
    trans=transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.ToTensor(),
        ])

    train_dataset = dataset.FileListDataset(
        '/home/hrzhang/projects/SSL-Backdoor/poison-generation/data/clean/train/clean_filelist_5subset.txt',
        trans)

    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.num_workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    return train_loader


def create_finetune_dataloader(args):
    trans = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([loader.GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    ])

    train_dataset = dataset.FileListDataset(
        '/home/hrzhang/projects/SSL-Backdoor/poison-generation/data/clean/train/finetune_train.txt',
        trans)

    val_dataset = dataset.FileListDataset(
        '/home/hrzhang/projects/SSL-Backdoor/poison-generation/data/clean/train/finetune_val.txt',
        val_transform)

    test_dataset = dataset.FileListDataset(
        '/home/hrzhang/projects/SSL-Backdoor/poison-generation/data/clean/train/finetune_test.txt',
        val_transform)


    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True)

    return train_loader,val_loader,test_loader
