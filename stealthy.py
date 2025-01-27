import torch
import numpy as np
from piq import fsim
from torchmetrics.image.fid import FrechetInceptionDistance
import lpips

import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from PIL import Image
import numpy as np

import torch.nn.functional as F
import torch.nn as nn
from torchvision.transforms import ToTensor
from datasets.CTRL.utils.frequency import PoisonFre

import copy,os,random,pilgram

test_transform_cifar10 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])


test_transform_stl10 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.44087798, 0.42790666, 0.38678814], [0.25507198, 0.24801506, 0.25641308])])


class BadEncoderTestBackdoor(Dataset):

    def __init__(self, numpy_file, transform=None):
        """
        Args:
            numpy_file (string): Path to the numpy file.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.input_array = np.load(numpy_file)
        self.data = self.input_array['x']
        self.targets = self.input_array['y']

        self.trigger_input_array = np.load('trigger/trigger_pt_white_21_10_ap_replace.npz')

        self.trigger_patch_list = self.trigger_input_array['t']
        self.trigger_mask_list = self.trigger_input_array['tm']


        self.test_transform = transform

        # state_dict = torch.load(trigger_file, map_location=torch.device('cpu'))
        # self.net = U_Net_tiny(img_ch=3,output_ch=3)
        # self.net.load_state_dict(state_dict['model_state_dict'])
        # self.net=self.net.eval()

        # self.filter = torch.load('trigger/filter.pt', map_location=torch.device('cpu'))

    def __getitem__(self,index):
        img = copy.deepcopy(self.data[index])
        clean_img = copy.deepcopy(self.data[index])
        ###########################
        ### SSLBKD ###


        ###########################
        ### for ins filter only ###

        # image_pil = Image.fromarray(img)
        # # filtered_image_pil = pilgram.xpro2(image_pil)
        # filtered_image_pil = pilgram.kelvin(image_pil)

        # img_backdoor =self.test_transform(filtered_image_pil)

        ###########################

        # img[:] =img * self.trigger_mask_list[0] + self.trigger_patch_list[0][:]
        # img_backdoor =self.test_transform(Image.fromarray(img))


        ###########################
        # for ctrl only
        # trans=transforms.Compose([
        #         transforms.ToTensor(),
        #     ])

        # image_pil = Image.fromarray(img)
        # tensor_image = trans(image_pil)

        # base_image=tensor_image.unsqueeze(0)
        # poison_frequency_agent = PoisonFre('args',32, [1,2], 32, [15,31],  False,  True)

        # x_tensor,_ = poison_frequency_agent.Poison_Frequency_Diff(base_image,0, 100.0)
        # img_backdoor = x_tensor.squeeze()

        # # img_backdoor = np.clip(img_backdoor, 0, 1) #限制颜色范围在0-1

        # img_backdoor = self.test_transform(img_backdoor.permute(1,2,0).detach().numpy())


        ########################


        img = Image.fromarray(img)
        img_backdoor =self.test_transform(img)

        clean_img = self.test_transform(clean_img)

        return img_backdoor, clean_img


    def __len__(self):
        return self.data.shape[0]
import torch
import numpy as np
# from skimage.metrics import structural_similarity as ssim
from pytorch_ssim import SSIM
# from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips
from torchmetrics.image import PeakSignalNoiseRatio
import kornia.augmentation as A
from optimize_filter.tiny_network import U_Net_tiny


psnr = PeakSignalNoiseRatio()

ssim = SSIM()

# 创建LPIPS模型
loss_fn = lpips.LPIPS(net='alex')

# def denormalize(tensor, mean, std):
#     """
#     反标准化函数：将标准化的张量恢复到原始像素值范围。
#     """
#     for t, m, s in zip(tensor, mean, std):
#         t.mul_(s).add_(m)
#     return tensor

def denormalize(tensor, mean, std):
    """
    反标准化函数：将标准化的张量恢复到原始像素值范围。
    """
    for t, m, s in zip(tensor, mean, std):
        t = t * s + m  # 非就地操作
    return tensor
def calculate_metrics(dataloader, mean, std):
    """
    计算SSIM, PSNR和LPIPS指标

    参数:
    - dataloader: DataLoader - 数据加载器，提供img_backdoor和clean_img对
    - mean: list - 标准化时使用的均值
    - std: list - 标准化时使用的标准差

    返回:
    - avg_ssim: float - 平均SSIM
    - avg_psnr: float - 平均PSNR
    - avg_lpips: float - 平均LPIPS
    """
    ssim_scores = []
    psnr_scores = []
    lpips_scores = []

    for img_backdoor, clean_img in dataloader:
        # 反标准化图像
        img_backdoor_denorm = denormalize(img_backdoor.clone(), mean, std)
        clean_img_denorm = denormalize(clean_img.clone(), mean, std)

        # 逐个样本计算SSIM和PSNR
        # img_b = img_backdoor_denorm[i].numpy().transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
        # img_c = clean_img_denorm[i].numpy().transpose(1, 2, 0)
        img_b = img_backdoor_denorm
        img_c = clean_img_denorm
        # 计算SSIM，指定较小的win_size并设置channel_axis
        ssim_value = ssim(img_b, img_c)
        ssim_scores.append(ssim_value)

        # 计算PSNR
        psnr_value = psnr(img_b, img_c)
        psnr_scores.append(psnr_value)

        # 计算LPIPS
        lpips_value = loss_fn(img_backdoor, clean_img)
        lpips_scores.extend(lpips_value.detach().cpu().numpy())

    # 计算平均值
    avg_ssim = np.mean(ssim_scores)
    avg_psnr = np.mean(psnr_scores)
    avg_lpips = np.mean(lpips_scores)

    return avg_ssim, avg_psnr, avg_lpips

def calculate_metrics_wanet(dataloader, mean, std):
    class ProbTransform(torch.nn.Module):
        def __init__(self, f, p=1):
            super(ProbTransform, self).__init__()
            self.f = f
            self.p = p

        def forward(self, x):  # , **kwargs):
            if random.random() < self.p:
                return self.f(x)
            else:
                return x

    class PostTensorTransform(torch.nn.Module):
        def __init__(self):
            super(PostTensorTransform, self).__init__()
            self.random_crop = ProbTransform(
                A.RandomCrop((32, 32), padding=5), p=0.8
            )
            self.random_rotation = ProbTransform(A.RandomRotation(10), p=0.5)


        def forward(self, x):
            for module in self.children():
                x = module(x)
            return x
    ssim_scores = []
    psnr_scores = []
    lpips_scores = []

    for img_backdoor, clean_img in dataloader:
        ############## add filter to backdoor img wanet
        input_height=32
        grid_rescale=1
        s=0.5
        k=4
        num_bd = img_backdoor.shape[0] // 2
        num_cross = num_bd
        ins = torch.rand(1, 2, k, k) * 2 - 1
        ins = ins / torch.mean(torch.abs(ins))
        noise_grid = (
            F.upsample(ins, size=input_height, mode="bicubic", align_corners=True)
            .permute(0, 2, 3, 1)
        )
        array1d = torch.linspace(-1, 1, steps=input_height)
        x, y = torch.meshgrid(array1d, array1d)
        identity_grid = torch.stack((y, x), 2)[None, ...]

        grid_temps = (identity_grid + s * noise_grid / input_height) * grid_rescale
        grid_temps = torch.clamp(grid_temps, -1, 1)
        transforms = PostTensorTransform()

        ins = torch.rand(num_cross, input_height, input_height, 2) * 2 - 1
        grid_temps2 = grid_temps.repeat(num_cross, 1, 1, 1) + ins / input_height
        grid_temps2 = torch.clamp(grid_temps2, -1, 1)

        inputs_bd = F.grid_sample(img_backdoor[:num_bd], grid_temps.repeat(num_bd, 1, 1, 1), align_corners=True)

        inputs_cross = F.grid_sample(img_backdoor[num_bd : (num_bd + num_cross)], grid_temps2, align_corners=True)

        img_backdoor = torch.cat([inputs_bd, inputs_cross], dim=0)

        img_backdoor_denorm = denormalize(img_backdoor.clone(), mean, std)
        clean_img_denorm = denormalize(clean_img.clone(), mean, std)

        # 逐个样本计算SSIM和PSNR
        # img_b = img_backdoor_denorm[i].numpy().transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
        # img_c = clean_img_denorm[i].numpy().transpose(1, 2, 0)
        img_b = img_backdoor_denorm
        img_c = clean_img_denorm
        # 计算SSIM，指定较小的win_size并设置channel_axis
        ssim_value = ssim(img_b, img_c)
        ssim_scores.append(ssim_value)

        # 计算PSNR
        psnr_value = psnr(img_b, img_c)
        psnr_scores.append(psnr_value)

        # 计算LPIPS
        lpips_value = loss_fn(img_backdoor, clean_img)
        lpips_scores.extend(lpips_value.detach().cpu().numpy())

    # 计算平均值
    avg_ssim = np.mean(ssim_scores)
    avg_psnr = np.mean(psnr_scores)
    avg_lpips = np.mean(lpips_scores)

    return avg_ssim, avg_psnr, avg_lpips

def clamp_batch_images(batch_images, mean, std):


    # 确保均值和标准差列表长度与通道数匹配
    num_channels =batch_images.shape[1]
    if len(mean) != num_channels or len(std) != num_channels:
        raise ValueError("The length of mean and std must match the number of channels")

    # 创建一个相同形状的张量用于存放裁剪后的图像

    clamped_images = torch.empty_like(batch_images)

    # 对每个通道分别进行裁剪
    for channel in range(batch_images.shape[1]):
        min_val = (0 - mean[channel]) / std[channel]
        max_val = (1 - mean[channel]) / std[channel]
        clamped_images[:, channel, :, :] = torch.clamp(batch_images[:, channel, :, :], min=min_val, max=max_val)

    return clamped_images

def calculate_metrics_inactive(dataloader, mean, std, trigger_file):
    ssim_scores = []
    psnr_scores = []
    lpips_scores = []

    for img_backdoor, clean_img in dataloader:
        state_dict = torch.load(trigger_file)
        filter = U_Net_tiny(img_ch=3,output_ch=3)
        filter.load_state_dict(state_dict['model_state_dict'])
        filter=filter.eval()
        img_backdoor=filter(img_backdoor)
        img_backdoor= clamp_batch_images(img_backdoor,mean,std)

        img_backdoor_denorm = denormalize(img_backdoor.clone(), mean, std)
        clean_img_denorm = denormalize(clean_img.clone(), mean, std)

        # 逐个样本计算SSIM和PSNR
        # img_b = img_backdoor_denorm[i].numpy().transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
        # img_c = clean_img_denorm[i].numpy().transpose(1, 2, 0)
        img_b = img_backdoor_denorm
        img_c = clean_img_denorm
        # 计算SSIM，指定较小的win_size并设置channel_axis
        ssim_value = ssim(img_b, img_c)
        ssim_scores.append(ssim_value.detach().cpu().numpy())

        # 计算PSNR
        psnr_value = psnr(img_b, img_c)
        psnr_scores.append(psnr_value.detach().cpu().numpy())

        # 计算LPIPS
        lpips_value = loss_fn(img_backdoor, clean_img)
        lpips_scores.extend(lpips_value.detach().cpu().numpy())

    # 计算平均值
    avg_ssim = np.mean(ssim_scores)
    avg_psnr = np.mean(psnr_scores)
    avg_lpips = np.mean(lpips_scores)

    return avg_ssim, avg_psnr, avg_lpips


def calculate_all_metrics(dataloader, mean, std, trigger_file, device='cuda'):
    fsim_scores = []
    lpips_scores = []
    fid_metric = FrechetInceptionDistance(feature=2048).to(device)  # FID 计算工具
    loss_fn = lpips.LPIPS(net='vgg').to(device)  # VGG Perceptual Loss 工具

    # 加载过滤器模型
    state_dict = torch.load(trigger_file)
    filter = U_Net_tiny(img_ch=3, output_ch=3).to(device)
    filter.load_state_dict(state_dict['model_state_dict'])
    filter = filter.eval()

    for img_backdoor, clean_img in dataloader:
        # 将数据加载到 GPU
        img_backdoor = img_backdoor.to(device)
        clean_img = clean_img.to(device)

        # 通过模型处理后门图像
        img_backdoor = filter(img_backdoor)
        img_backdoor = clamp_batch_images(img_backdoor, mean, std)

        # 去归一化并批量处理
        img_backdoor_denorm = denormalize(img_backdoor.clone(), mean, std)
        clean_img_denorm = denormalize(clean_img.clone(), mean, std)

        # 转换为 uint8 类型
        img_backdoor_denorm = (img_backdoor_denorm * 255).clamp(0, 255).to(torch.uint8)
        clean_img_denorm = (clean_img_denorm * 255).clamp(0, 255).to(torch.uint8)

        # 计算 FSIM (支持批量计算)
        fsim_batch = fsim(img_backdoor_denorm, clean_img_denorm, data_range=255)  # 批量计算
        fsim_scores.extend(fsim_batch.cpu().numpy())

        # 更新 FID 批量
        fid_metric.update(img_backdoor_denorm, real=False)
        fid_metric.update(clean_img_denorm, real=True)

        # 计算 VGG Perceptual Loss (支持批量)
        lpips_batch = loss_fn(img_backdoor, clean_img)
        lpips_scores.extend(lpips_batch.detach().cpu().numpy())

    # 计算最终结果
    avg_fsim = np.mean(fsim_scores)
    fid_score = fid_metric.compute().item()
    avg_lpips = np.mean(lpips_scores)

    return {
        'FSIM': avg_fsim,
        'FID': fid_score,
        'Perceptual_Loss': avg_lpips
    }


data = [
    ('cifar10', 'stl10', 'output/cifar10/stl10_backdoored_encoder/2023-12-29-13:14:52/unet_filter_200_trained.pt'),
    ('cifar10', 'svhn', 'output/cifar10/svhn_backdoored_encoder/2023-12-26-13:50:32/unet_filter_200_trained.pt'),
    ('cifar10', 'gtsrb', 'output/cifar10/gtsrb_backdoored_encoder/2024-01-28-13:17:00/unet_filter_200_trained.pt'),
    ('stl10', 'cifar10', 'output/stl10/cifar10_backdoored_encoder/2023-12-25-17:10:56/unet_filter_200_trained.pt'),
    ('stl10', 'gtsrb', 'output/stl10/gtsrb_backdoored_encoder/2023-12-25-16:45:06/unet_filter_200_trained.pt'),
    ('stl10', 'svhn', 'output/stl10/svhn_backdoored_encoder/2024-01-31-09:39:14/unet_filter_200_trained.pt')
]

for pre, down, trigger_file in data:
    # 设置均值和标准差
    if pre == 'cifar10':
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        transform = test_transform_cifar10
    else:
        mean = [0.44087798, 0.42790666, 0.38678814]
        std = [0.25507198, 0.24801506, 0.25641308]

    # 创建测试 DataLoader
    bd_test = BadEncoderTestBackdoor(f'data/{down}/test.npz', transform)
    dataloader = DataLoader(bd_test, batch_size=4096, shuffle=False)

    # 计算指标
    metrics = calculate_all_metrics(dataloader, mean, std, trigger_file)

    # 打印结果
    print('*' * 25)
    print('Pretrained Dataset:', pre)
    print('Downstream Dataset:', down)
    print(f"FSIM: {metrics['FSIM']:.4f}")
    print(f"FID: {metrics['FID']:.4f}")
    print(f"Perceptual Loss: {metrics['Perceptual_Loss']:.4f}")
    print('*' * 25)
