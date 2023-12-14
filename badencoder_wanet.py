import argparse

import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import torch,lpips
import torch.nn as nn
import torch.nn.functional as F

from models import get_encoder_architecture_usage
from datasets import get_shadow_dataset

from datetime import datetime
from optimize_filter.loss import *
import os,random
import kornia.augmentation as A

now = datetime.now()
print("当前时间：", now.strftime("%Y-%m-%d %H:%M:%S"))


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


def train(backdoored_encoder, clean_encoder, data_loader, train_optimizer, args):

    backdoored_encoder.train()

    for module in backdoored_encoder.modules():
    # print(module)
        if isinstance(module, nn.BatchNorm2d):
            if hasattr(module, 'weight'):
                module.weight.requires_grad_(False)
            if hasattr(module, 'bias'):
                module.bias.requires_grad_(False)
            module.eval()

    clean_encoder.eval()

    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    total_loss_0, total_loss_1, total_loss_2 = 0.0, 0.0, 0.0

    for img_clean, img_backdoor_list, reference_list,reference_aug_list, img_trans in train_bar:
        img_clean = img_clean.cuda(non_blocking=True)
        img_trans = img_trans.cuda(non_blocking=True)

        reference_cuda_list, reference_aug_cuda_list, img_backdoor_cuda_list = [], [], []
        for reference in reference_list:
            reference_cuda_list.append(reference.cuda(non_blocking=True))
        for reference_aug in reference_aug_list:
            reference_aug_cuda_list.append(reference_aug.cuda(non_blocking=True))
        for img_backdoor in img_backdoor_list:
            img_backdoor_cuda_list.append(img_backdoor.cuda(non_blocking=True))

        clean_feature_reference_list = []

        with torch.no_grad():
            clean_feature_raw = clean_encoder(img_clean)
            clean_feature_raw = F.normalize(clean_feature_raw, dim=-1)
            for img_reference in reference_cuda_list:
                clean_feature_reference = clean_encoder(img_reference)
                clean_feature_reference = F.normalize(clean_feature_reference, dim=-1)
                clean_feature_reference_list.append(clean_feature_reference)


        feature_raw = backdoored_encoder(img_clean)
        feature_raw = F.normalize(feature_raw, dim=-1)

        feature_backdoor_list = []
        for img_backdoor in img_backdoor_cuda_list:
            ############## add filter to backdoor img
            # filter.eval()
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
                .cuda()
            )
            array1d = torch.linspace(-1, 1, steps=input_height)
            x, y = torch.meshgrid(array1d, array1d)
            identity_grid = torch.stack((y, x), 2)[None, ...].cuda()

            grid_temps = (identity_grid + s * noise_grid / input_height) * grid_rescale
            grid_temps = torch.clamp(grid_temps, -1, 1)
            transforms = PostTensorTransform().cuda()

            ins = torch.rand(num_cross, input_height, input_height, 2).cuda() * 2 - 1
            grid_temps2 = grid_temps.repeat(num_cross, 1, 1, 1) + ins / input_height
            grid_temps2 = torch.clamp(grid_temps2, -1, 1)

            inputs_bd = F.grid_sample(img_backdoor[:num_bd], grid_temps.repeat(num_bd, 1, 1, 1), align_corners=True)

            inputs_cross = F.grid_sample(img_backdoor[num_bd : (num_bd + num_cross)], grid_temps2, align_corners=True)

            total_inputs = torch.cat([inputs_bd, inputs_cross], dim=0)
            total_inputs = transforms(total_inputs)

            # img_backdoor = torch.clamp(img_backdoor, min=0, max=1)

            feature_backdoor = backdoored_encoder(img_backdoor)
            feature_backdoor = F.normalize(feature_backdoor, dim=-1) # 按照特征维度归一化特征，即每个特征的平方和为1
            feature_backdoor_list.append(feature_backdoor)

        feature_reference_list = []
        for img_reference in reference_cuda_list:
            feature_reference = backdoored_encoder(img_reference)
            feature_reference = F.normalize(feature_reference, dim=-1)
            feature_reference_list.append(feature_reference)

        feature_reference_aug_list = []
        for img_reference_aug in reference_aug_cuda_list:
            feature_reference_aug = backdoored_encoder(img_reference_aug)
            feature_reference_aug = F.normalize(feature_reference_aug, dim=-1)
            feature_reference_aug_list.append(feature_reference_aug)

        loss_0_list, loss_1_list = [], []
        for i in range(len(feature_reference_list)):
            loss_0_list.append(- torch.sum(feature_backdoor_list[i] * feature_reference_list[i], dim=-1).mean()) # 计算两者之间的负点积的平均值，点积越大，表示特征向量之间的相似性越高。通过最小化这个负值，模型被训练以使得带有后门的图片的特征表示与参考图片尽可能相似。
            loss_1_list.append(- torch.sum(feature_reference_aug_list[i] * clean_feature_reference_list[i], dim=-1).mean())

        loss_2 = - torch.sum(feature_raw * clean_feature_raw, dim=-1).mean()

        loss_0 = sum(loss_0_list)/len(loss_0_list)
        loss_1 = sum(loss_1_list)/len(loss_1_list)

        loss = loss_0 + args.lambda1 * loss_1 + args.lambda2 * loss_2

        train_optimizer.zero_grad()
        loss.backward()
        # loss.backward()
        train_optimizer.step()


        total_num += data_loader.batch_size
        total_loss += loss.item() * data_loader.batch_size
        total_loss_0 += loss_0.item() * data_loader.batch_size
        total_loss_1 += loss_1.item() * data_loader.batch_size
        total_loss_2 += loss_2.item() * data_loader.batch_size
        train_bar.set_description('Train Epoch: [{}/{}], lr: {:.6f}, Loss: {:.6f}, Loss0: {:.6f}, Loss1: {:.6f},  Loss2: {:.6f}'.format(epoch, args.epochs, train_optimizer.param_groups[0]['lr'], total_loss / total_num,  total_loss_0 / total_num , total_loss_1 / total_num,  total_loss_2 / total_num))


    return total_loss / total_num


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Finetune the encoder to get the backdoored encoder')
    parser.add_argument('--batch_size', default=256, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate in SGD')
    parser.add_argument('--lambda1', default=1.0, type=np.float64, help='value of labmda1')
    parser.add_argument('--lambda2', default=1.0, type=np.float64, help='value of labmda2')
    parser.add_argument('--epochs', default=200, type=int, help='Number of sweeps over the shadow dataset to inject the backdoor')

    parser.add_argument('--reference_file', default='', type=str, help='path to the reference inputs')
    parser.add_argument('--trigger_file', default='', type=str, help='path to the trigger')
    parser.add_argument('--shadow_dataset', default='cifar10', type=str,  help='shadow dataset')
    parser.add_argument('--pretrained_encoder', default='', type=str, help='path to the clean encoder used to finetune the backdoored encoder')
    parser.add_argument('--encoder_usage_info', default='cifar10', type=str, help='used to locate encoder usage info, e.g., encoder architecture and input normalization parameter')

    parser.add_argument('--results_dir', default='', type=str, metavar='PATH', help='path to save the backdoored encoder')

    parser.add_argument('--seed', default=100, type=int, help='which seed the code runs on')
    parser.add_argument('--gpu', default='1', type=str, help='which gpu the code runs on')
    parser.add_argument('--pretraining_dataset', type=str, default='cifar10')

    parser.add_argument('--timestamp', type=str)

    parser.add_argument('--hue_hsv', type=float, default=1)
    parser.add_argument('--saturation_hsv', type=float, default=1)
    parser.add_argument('--value_hsv', type=float, default=1)
    parser.add_argument('--lightness', type=float, default=1)

    parser.add_argument('--psnr', type=float, default=0.025)
    parser.add_argument('--loss0', type=float, default=1)
    parser.add_argument('--color', type=float, default=0)

    args = parser.parse_args()

    # os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

    # Specify the pre-training data directory
    args.data_dir = f'./data/{args.shadow_dataset.split("_")[0]}/'
    args.knn_k = 200
    args.knn_t = 0.5
    args.reference_label = 0
    print(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Create the Pytorch Datasets, and create the data loader for the training set
    # memory_data, test_data_clean, and test_data_backdoor are used to monitor the finetuning process. They are not reqruied by our BadEncoder
    shadow_data, memory_data, test_data_clean, test_data_backdoor = get_shadow_dataset(args)

    clean_model = get_encoder_architecture_usage(args).cuda()
    model = get_encoder_architecture_usage(args).cuda()

    # Create the extra data loaders for testing purpose and define the optimizer
    print("Optimizer: SGD")

    if args.encoder_usage_info == 'cifar10' or args.encoder_usage_info == 'stl10':
        # note that the following three dataloaders are used to monitor the finetune of the pre-trained encoder, they are not required by our BadEncoder. They can be ignored if you do not need to monitor the finetune of the pre-trained encoder
        memory_loader = DataLoader(memory_data, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
        test_loader_clean = DataLoader(test_data_clean, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
        test_loader_backdoor = DataLoader(test_data_backdoor, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
        optimizer = torch.optim.SGD(model.f.parameters(), lr=args.lr, weight_decay=5e-4, momentum=0.9)
    else:
        optimizer = torch.optim.SGD(model.visual.parameters(), lr=args.lr, weight_decay=5e-4, momentum=0.9)

    # Initialize the BadEncoder and load the pretrained encoder
    if args.pretrained_encoder != '':
        print(f'load the clean model from {args.pretrained_encoder}')
        if args.encoder_usage_info == 'cifar10' or args.encoder_usage_info == 'stl10':
            checkpoint = torch.load(args.pretrained_encoder)
            clean_model.load_state_dict(checkpoint['state_dict'])
            model.load_state_dict(checkpoint['state_dict'])
        elif args.encoder_usage_info == 'imagenet' or args.encoder_usage_info == 'CLIP':
            checkpoint = torch.load(args.pretrained_encoder)
            clean_model.visual.load_state_dict(checkpoint['state_dict'])
            model.visual.load_state_dict(checkpoint['state_dict'])
        else:
            raise NotImplementedError()



    # if args.encoder_usage_info == 'cifar10' or args.encoder_usage_info == 'stl10':
    #     # check whether the pre-trained encoder is loaded successfully or not
    #     test_acc_1 = test(model.f, memory_loader, test_loader_clean, test_loader_backdoor,0, args,net)
    #     print('initial test acc: {}'.format(test_acc_1))

    # training loop
    for epoch in range(1, args.epochs + 1):
        print("=================================================")

        train_loader = DataLoader(shadow_data, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)

        if args.encoder_usage_info == 'cifar10' or args.encoder_usage_info == 'stl10':
            train_loss = train(model.f, clean_model.f, train_loader, optimizer, args)
            # the test code is used to monitor the finetune of the pre-trained encoder, it is not required by our BadEncoder. It can be ignored if you do not need to monitor the finetune of the pre-trained encoder
            # _ = test(model.f, memory_loader, test_loader_clean, test_loader_backdoor, epoch, args,net)
        elif args.encoder_usage_info == 'imagenet' or args.encoder_usage_info == 'CLIP':
            train_loss = train(model.visual, clean_model.visual, train_loader, optimizer, args)
        else:
            raise NotImplementedError()

        # Save the BadEncoder
        if epoch % 25 == 0:
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict(),}, args.results_dir + '/model_' + str(epoch) + '.pth')
    #     torch.save({'model_state_dict': net.state_dict()}, args.results_dir + f'/{args.timestamp}/unet_filter_trained_ssim{ssim:.4f}_psnr{psnr:.2f}_lp{lp:.4f}_wd{wd:.3f}.pt')

        # Save the intermediate checkpoint
        # torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict(),}, args.results_dir + '/model_last.pth')

        now = datetime.now()
        print("当前时间：", now.strftime("%Y-%m-%d %H:%M:%S"))