import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class PairedImageDataset(Dataset):
    def __init__(self, root_issba, root_train, transform=None):
        self.root_issba = root_issba
        self.root_train = root_train
        self.transform = transform

        # 获取train_issba目录下的所有文件路径
        self.issba_images = []
        for subdir, _, files in os.walk(root_issba):
            for file in files:
                if file.endswith(('.jpg', '.jpeg', '.png', '.JPEG', '.png')):
                    self.issba_images.append(os.path.join(subdir, file))

    def __len__(self):
        return len(self.issba_images)

    def __getitem__(self, idx):
        issba_image_path = self.issba_images[idx]
        image_name = os.path.basename(issba_image_path)  # 获取文件名

        # 获取对应的train文件夹下的图片路径
        train_image_path = issba_image_path.replace(self.root_issba, self.root_train)

        # 加载图像
        issba_image = Image.open(issba_image_path).convert("RGB")
        train_image = Image.open(train_image_path).convert("RGB")

        if self.transform:
            issba_image = self.transform(issba_image)
            train_image = self.transform(train_image)

        return issba_image, train_image


import torch
import numpy as np
# from skimage.metrics import structural_similarity as ssim
from pytorch_ssim import SSIM
# from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips
from torchmetrics.image import PeakSignalNoiseRatio
import kornia.augmentation as A
from optimize_filter.tiny_network import U_Net_tiny

# 检查CUDA设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 将模型和损失函数移动到CUDA
psnr = PeakSignalNoiseRatio().to(device)
ssim = SSIM().to(device)
loss_fn = lpips.LPIPS(net='alex').to(device)

def denormalize(tensor, mean, std):
    """
    反标准化函数：将标准化的张量恢复到原始像素值范围。
    """
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def calculate_metrics(dataloader, mean, std, device):
    """
    计算SSIM, PSNR和LPIPS指标

    参数:
    - dataloader: DataLoader - 数据加载器，提供img_backdoor和clean_img对
    - mean: list - 标准化时使用的均值
    - std: list - 标准化时使用的标准差
    - device: torch.device - 计算设备（CPU或CUDA）

    返回:
    - avg_ssim: float - 平均SSIM
    - avg_psnr: float - 平均PSNR
    - avg_lpips: float - 平均LPIPS
    """
    ssim_scores = []
    psnr_scores = []
    lpips_scores = []

    for img_backdoor, clean_img in dataloader:
        # 移动到CUDA设备
        img_backdoor = img_backdoor.to(device)
        clean_img = clean_img.to(device)

        # 反标准化图像
        img_backdoor_denorm = denormalize(img_backdoor.clone(), mean, std)
        clean_img_denorm = denormalize(clean_img.clone(), mean, std)

        # 计算SSIM
        ssim_value = ssim(img_backdoor_denorm, clean_img_denorm)
        ssim_scores.append(ssim_value.item())

        # 计算PSNR
        psnr_value = psnr(img_backdoor_denorm, clean_img_denorm)
        psnr_scores.append(psnr_value.item())

        # 计算LPIPS
        lpips_value = loss_fn(img_backdoor_denorm, clean_img_denorm)
        lpips_scores.extend(lpips_value.detach().cpu().numpy())

    # 计算平均值
    avg_ssim = np.mean(ssim_scores)
    avg_psnr = np.mean(psnr_scores)
    avg_lpips = np.mean(lpips_scores)

    return avg_ssim, avg_psnr, avg_lpips

data = ['svhn','gtsrb','stl10']
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

for down in data:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    bd_test = PairedImageDataset(f'data/{down}/test_224_issba', f'data/{down}/test_224',train_transform)
    dataloader = DataLoader(bd_test, batch_size=32, shuffle=False)
    # 使用 DataLoader 计算指标
    avg_ssim, avg_psnr, avg_lpips = calculate_metrics(dataloader, mean, std,device)
    print('*'*25)
    print('down:', down)
    print(f'Average SSIM: {avg_ssim}')
    print(f'Average PSNR: {avg_psnr} dB')
    print(f'Average LPIPS: {avg_lpips}')
    print('*'*25)