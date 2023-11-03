import sys
sys.path.append("..")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR

import numpy as np

import matplotlib.pyplot as plt
from pytorch_ssim import SSIM
from datetime import datetime
# from piq import ssim, SSIMLoss
from tqdm import tqdm

from torchvision import transforms
import numpy as np

import os,random

import moco.loader as loader
import moco.dataset as dataset
from network import U_Net

def seed_torch(seed=1029):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
    #torch.use_deterministic_algorithms(True)  # 有检查操作，看下文区别

# seed_torch()
timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

# 检查是否有可用的 GPU
if torch.cuda.is_available():
    # 获取当前运行的 GPU 编号
    current_gpu = torch.cuda.current_device()
    print(f"当前运行的 GPU 编号为: {current_gpu}")
else:
    print("没有可用的 GPU。")

# if args.aug_plus:
#     # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
#     augmentation = [
#         transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
#         transforms.RandomApply([
#             transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
#         ], p=0.8),
#         transforms.RandomGrayscale(p=0.2),
#         transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         normalize,
#     ]
# else:
#     # MoCo v1's aug: the same as InstDisc https://arxiv.org/abs/1805.01978
#     augmentation = [
#         transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
#         transforms.RandomGrayscale(p=0.2),
#         transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         normalize,
#     ]
def create_data_loader():
    # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
    augmentation = [
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.RandomGrayscale(p=1.0),
        loader.GaussianBlur([.1, 2.]),
        transforms.ToTensor(),
    ]

    trans=transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.ToTensor()])

    # Filelist loading
    train_dataset = dataset.FileListDataset(
        '/home/hrzhang/projects/SSL-Backdoor/poison-generation/data/clean/train/clean_filelist_20subset.txt',
        transforms.Compose(augmentation),
        trans)

    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=256, shuffle=(train_sampler is None),
        num_workers=2, pin_memory=True, sampler=train_sampler, drop_last=True)

    return train_loader


train_loader = create_data_loader()

filter_size = 7


# 设计滤镜：滤镜是一个3x3的卷积核，作为一个可学习的参数
filter = torch.nn.Parameter(torch.randn(3, 3, filter_size, filter_size, requires_grad=True).cuda())  # 修改滤波器形状并将其放在GPU上

# filter=torch.load('trigger/moco/filter_2023-09-16-13-25-37.pt')

# 优化滤镜：使用SGD算法优化滤波器的参数，使得输入图像与经过train_transform转换后的图像的差异最大
num_iterations = 500

optimizer = torch.optim.Adam([filter], lr=1e-2)

scheduler = StepLR(optimizer, step_size=200, gamma=0.1)

print('Start training...')
bar=tqdm(range(1, num_iterations+1))

for _ in bar:
    total_loss = 0.0
    for img_trans, img in train_loader:
        # 将输入图像和转换后的图像都放在GPU上
        img = img.cuda()
        img_trans = img_trans.cuda()
        # 将滤镜作用在输入图像上，得到扰动后的图像
        filter_img = F.conv2d(img, filter, padding=filter_size//2)  # 修改滤波器形状

        mse = nn.MSELoss()
        ssim = SSIM()

        # filter后的图片和经过transform后的mse和ssim，差距要尽可能大
        loss_mse1 = mse(filter_img, img_trans)
        loss_ssim1 = ssim(filter_img,img_trans)

        # filter后的图片和原图的mse和ssim，差距要尽可能小
        loss_mse2 = mse(filter_img, img)
        loss_ssim2 = ssim(filter_img, img)

        ############################ IMAGENET ############################
        loss = 10*(0.00001 * loss_mse2 + 50*(1 - loss_ssim2)) - (0.00001 * loss_mse1 + 50*(1 - loss_ssim1))
        # loss = 0.00001 * loss_mse2 + 50*(1 - loss_ssim2) - (0.00001 * loss_mse1 + 50*(1 - loss_ssim1))

        # 梯度清零
        optimizer.zero_grad()

        # 计算损失函数相对于滤波器参数的梯度
        loss.backward()

        torch.nn.utils.clip_grad_norm_([filter], max_norm=1.0)  # Adjust the value of max_norm if needed

        # 使用SGD更新滤波器参数
        optimizer.step()
        total_loss += loss.item()

    scheduler.step()
    # 计算平均损失
    avg_loss = total_loss / len(train_loader)

    # 输出当前epoch的loss
    bar.set_description(f"Loss: {avg_loss}, lr: {optimizer.param_groups[0]['lr']}, SSIM1: {loss_ssim1.item()}, MSE1: {loss_mse1.item()}, SSIM2: {loss_ssim2.item()}, MSE2: {loss_mse2.item()}")
    # bar.set_description(f"Loss: {avg_loss}, lr: {optimizer.param_groups[0]['lr']}, SSIM: {loss_ssim.item()}, MSE: {loss_mse.item()}")
    torch.save(filter, f'trigger/moco/filter_{timestamp}.pt')