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
])



def create_data_loader(args):
    # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
    # augmentation = [
    #     transforms.RandomResizedCrop(224),
    #     transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    #     transforms.RandomGrayscale(p=1.0),
    #     loader.GaussianBlur([.1, 2.]),
    #     transforms.ToTensor(),
    # ]

    trans=transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.ToTensor(),
        ])

    # Filelist loading
    train_dataset = dataset.FileListDataset(
        '/home/hrzhang/projects/SSL-Backdoor/poison-generation/data/clean/train/clean_filelist_5subset.txt',
        aug,trans)

    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.num_workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    return train_loader