import os
import argparse
import random

import torchvision
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import torch,lpips
import torch.nn as nn
import torch.nn.functional as F

from models import get_encoder_architecture_usage
from datasets import get_shadow_dataset
from evaluation import test
from optimize_filter.network import AttU_Net,U_Net
from optimize_filter.tiny_network import U_Net_tiny
from optimize_filter.utils import SinkhornDistance, Recorder, Loss_Tracker
from torch.nn import MSELoss,Sigmoid
from pytorch_ssim import SSIM
from torchmetrics.image import PeakSignalNoiseRatio
from datetime import datetime

# 获取当前时间
now = datetime.now()
print("当前时间：", now.strftime("%Y-%m-%d %H:%M:%S"))


# def filter_wd_loss(filter,img_clean,img_trans,psnr,ssim,loss_fn,WD,tracker,recorder):
#     # mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1).cuda()
#     # std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1).cuda()

#     filter_img = filter(img_clean)
#     # filter_img = filter_img * std + mean
#     # img_clean = img_clean * std + mean
#     # img_trans_unnorm = img_trans * std + mean
#     wd,_,_=WD(filter_img.view(filter_img.shape[0],-1),img_trans.view(img_trans.shape[0],-1))
#     # wd,_,_=WD(filter_img_unnorm.view(filter_img.shape[0],-1),img_trans_unnorm.view(img_trans.shape[0],-1))

#     loss_psnr = psnr(filter_img, img_clean)
#     loss_ssim = ssim(filter_img, img_clean)
#     d_list = loss_fn(filter_img,img_clean)
#     lp_loss=d_list.squeeze()

#     loss_sim = 1 - loss_ssim + 10 * lp_loss.mean() - 0.025 * loss_psnr
#     loss_far = recorder.cost * wd

#     loss = loss_sim - loss_far

#     print(f'\nloss:{loss},loss_sim:{loss_sim}, loss_far:{loss_far}, wd:{wd},ssim:{loss_ssim},lp:{lp_loss.mean()},psnr:{loss_psnr},cost:{recorder.cost}')

#     tracker.update(loss.item(),wd.item(),loss_ssim.item(),loss_psnr.item(),lp_loss.mean().item(),loss_sim.item(),loss_far.item())
#     return loss


def train(backdoored_encoder, clean_encoder, data_loader, train_optimizer, args, filter=None):
    # tracker.reset()
    # WD=SinkhornDistance(eps=0.1, max_iter=100)
    # ssim = SSIM()
    #     # self.loss_cmd = CMD()
    # loss_fn = lpips.LPIPS(net='alex').cuda()
    # psnr = PeakSignalNoiseRatio().cuda()
    # sig=Sigmoid()

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

    for img_clean, img_backdoor_list, reference_list,reference_aug_list in train_bar:
        img_clean = img_clean.cuda(non_blocking=True)
        # img_trans = img_trans.cuda(non_blocking=True)

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
            # img_backdoor=filter(img_backdoor)
            # img_backdoor=sig(img_backdoor)

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
        # feature_backdoor_list: backdoor encoder输出backdoored raw image（整个数据集的一部分）的特征
        # feature_reference_list: backdoor encoder输出reference image的特征
        # feature_reference_aug_list ：backdoor encoder输出reference image的augmentation的特征，包含的是参考图片的增强版本（例如旋转、缩放等）经过被攻击的编码器后的特征表示。
        # clean_feature_reference_list ：clean encoder输出reference image的特征 是同样的参考图片经过未被攻击的编码器后的特征表示。
        # 通过最小化 loss_1_list 中的负点积的平均值，模型被训练以确保即使在数据增强之后，被攻击的编码器和未被攻击的编码器产生的特征表示仍然保持一致性。
        # our effectiveness loss is smaller if the backdoored image encoder
        # 1) produces more similar feature vectors for the reference inputs and the inputs in the shadow dataset embedded with the trigger for each (target downstream task, target class) pair,
        # 2) produces more similar feature vectors for the reference inputs with the clean image encoder.
        # Our utility loss is smaller if the backdoored image encoder and the clean image encoder produce more similar feature vectors for each clean input in the shadow dataset.

        loss_2 = - torch.sum(feature_raw * clean_feature_raw, dim=-1).mean()
        #feature_raw 是原始干净图片通过被攻击的编码器后的特征表示。
        # clean_feature_raw 是同样的原始图片通过未被攻击的编码器后的特征表示。
        # loss_2 计算这两组特征表示的负点积的平均值。最小化 loss_2 的过程中，确保在没有后门触发的情况下，被攻击的编码器和未被攻击的编码器的输出特征表示保持一致。

        loss_0 = sum(loss_0_list)/len(loss_0_list)
        loss_1 = sum(loss_1_list)/len(loss_1_list)

        loss = loss_0 + args.lambda1 * loss_1 + args.lambda2 * loss_2

        train_optimizer.zero_grad()
        # loss.backward(retain_graph=True)
        loss.backward()
        train_optimizer.step()


        total_num += data_loader.batch_size
        total_loss += loss.item() * data_loader.batch_size
        total_loss_0 += loss_0.item() * data_loader.batch_size
        total_loss_1 += loss_1.item() * data_loader.batch_size
        total_loss_2 += loss_2.item() * data_loader.batch_size
        train_bar.set_description('Train Epoch: [{}/{}], lr: {:.6f}, Loss: {:.6f}, Loss0: {:.6f}, Loss1: {:.6f},  Loss2: {:.6f}'.format(epoch, args.epochs, train_optimizer.param_groups[0]['lr'], total_loss / total_num,  total_loss_0 / total_num , total_loss_1 / total_num,  total_loss_2 / total_num))
        # train_bar.set_description('Train Epoch: [{}/{}], lr: {:.6f}, Loss: {:.6f}, Loss0: {:.6f}, Loss1: {:.6f},  Loss2: {:.6f}, Loss_wd:{:.6f}'.format(epoch, args.epochs, train_optimizer.param_groups[0]['lr'], total_loss / total_num,  total_loss_0 / total_num , total_loss_1 / total_num,  total_loss_2 / total_num, wd_loss.item()))

    # avg_loss,wd,ssim,psnr,lp,sim,far = tracker.get_avg_loss()

    # if ssim >= args.ssim_threshold and psnr >= args.psnr_threshold and lp <= args.lp_threshold and wd >= recorder.best:
    #     torch.save({'model_state_dict': net.state_dict()}, args.results_dir + f'/{args.timestamp}/unet_filter_trained_ssim{ssim:.4f}_psnr{psnr:.2f}_lp{lp:.4f}_wd{wd:.3f}.pt')

    #     recorder.best = wd
    #     print('\n--------------------------------------------------')
    #     print(f"Updated !!! Best sim:{sim}, far:{far}, SSIM: {ssim}, psnr: {psnr}, lp: {lp}, Best WD: {wd}")
    #     print('--------------------------------------------------')
    #     recorder.cost_up_counter = 0
    #     recorder.cost_down_counter = 0


    # if ssim >= args.ssim_threshold and psnr >= args.psnr_threshold and lp <= args.lp_threshold:
    #     recorder.cost_up_counter += 1
    #     recorder.cost_down_counter = 0
    # else:
    #     recorder.cost_up_counter = 0
    #     recorder.cost_down_counter += 1

    # if recorder.cost_up_counter >= args.patience:
    #     recorder.cost_up_counter = 0
    #     print('\n--------------------------------------------------')
    #     print("Up cost from {} to {}".format(recorder.cost, recorder.cost * recorder.cost_multiplier_up))
    #     print('--------------------------------------------------')

    #     recorder.cost *= recorder.cost_multiplier_up
    #     if recorder.cost > args.max_cost:
    #         recorder.cost = args.max_cost
    #     recorder.cost_up_flag = True

    # elif recorder.cost_down_counter >= args.patience:
    #     recorder.cost_down_counter = 0
    #     print('\n--------------------------------------------------')
    #     print("Down cost from {} to {}".format(recorder.cost, recorder.cost / recorder.cost_multiplier_down))
    #     print('--------------------------------------------------')
    #     recorder.cost /= recorder.cost_multiplier_down
    #     if recorder.cost < args.min_cost:
    #         recorder.cost = args.min_cost
    #     recorder.cost_down_flag = True


    # # if args.use_feature:
    # print(f"Loss: {avg_loss}, SIM: {sim:.5f}, far:{far}, WD: {wd}, SSIM: {ssim:.5f}, pnsr:{psnr:.5f}, lp:{lp:.5f},  cost:{recorder.cost}")
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
    # parser.add_argument('--filter_path', default='', type=str, help='path to the filter trigger')
    # parser.add_argument('--ssim_threshold', type=float, default=0.90)
    # parser.add_argument('--psnr_threshold', type=float, default=25.0)
    # parser.add_argument('--lp_threshold', type=float, default=0.003)
    # parser.add_argument('--patience', type=int, default=3)
    # parser.add_argument('--init_cost', type=float, default=5e-3)
    # parser.add_argument('--cost_multiplier_up', type=float, default=1.2)
    # parser.add_argument('--cost_multiplier_down', type=float, default=1.5)
    parser.add_argument('--timestamp', type=str)
    # parser.add_argument('--max_cost', type=float, default=1e-2)
    # parser.add_argument('--min_cost', type=float, default=1e-3)

    args = parser.parse_args()

    # # Set the seed and determine the GPU
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu
    # # print('gpu:',args.gpu)
    # random.seed(args.seed)
    # os.environ['PYTHONHASHSEED'] = str(args.seed)
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

    # Specify the pre-training data directory
    args.data_dir = f'./data/{args.shadow_dataset.split("_")[0]}/'
    args.knn_k = 200
    args.knn_t = 0.5
    args.reference_label = 0
    print(args)

    # Create the Pytorch Datasets, and create the data loader for the training set
    # memory_data, test_data_clean, and test_data_backdoor are used to monitor the finetuning process. They are not reqruied by our BadEncoder
    shadow_data, memory_data, test_data_clean, test_data_backdoor = get_shadow_dataset(args)
    train_loader = DataLoader(shadow_data, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)

    clean_model = get_encoder_architecture_usage(args).cuda()
    model = get_encoder_architecture_usage(args).cuda()

    # Create the extra data loaders for testing purpose and define the optimizer
    print("Optimizer: SGD")

    if args.encoder_usage_info == 'cifar10' or args.encoder_usage_info == 'stl10' or args.encoder_usage_info == 'gtsrb':
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
        if args.encoder_usage_info == 'cifar10' or args.encoder_usage_info == 'stl10' or args.encoder_usage_info == 'gtsrb':
            checkpoint = torch.load(args.pretrained_encoder)
            clean_model.load_state_dict(checkpoint['state_dict'])
            model.load_state_dict(checkpoint['state_dict'])
        elif args.encoder_usage_info == 'imagenet' or args.encoder_usage_info == 'CLIP':
            checkpoint = torch.load(args.pretrained_encoder)
            clean_model.visual.load_state_dict(checkpoint['state_dict'])
            model.visual.load_state_dict(checkpoint['state_dict'])
        else:
            raise NotImplementedError()

    # if args.encoder_usage_info == 'imagenet':
    # state_dict = torch.load(args.trigger_file, map_location=torch.device('cuda:0'))
    # net = U_Net_tiny(img_ch=3,output_ch=3)
    # net.load_state_dict(state_dict['model_state_dict'])
    # net=net.cuda().eval()
        # optimizer_wd = torch.optim.Adam(list(net.parameters()), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
        # recorder=Recorder(args)
        # tracker=Loss_Tracker(['loss', 'wd', 'ssim', 'psnr', 'lp', 'sim', 'far','color'])

    if args.encoder_usage_info == 'cifar10' or args.encoder_usage_info == 'stl10' or args.encoder_usage_info == 'gtsrb':
        # check whether the pre-trained encoder is loaded successfully or not
        test_acc_1 = test(model.f, memory_loader, test_loader_clean, test_loader_backdoor,0, args)
        print('initial test acc: {}'.format(test_acc_1))

    # training loop
    for epoch in range(1, args.epochs + 1):
        print("=================================================")
        if args.encoder_usage_info == 'cifar10' or args.encoder_usage_info == 'stl10' or args.encoder_usage_info == 'gtsrb':
            # train_loss = train(model.f, clean_model.f, train_loader, optimizer, args, net, optimizer_wd, tracker,recorder)
            train_loss = train(model.f, clean_model.f, train_loader, optimizer, args)
            # the test code is used to monitor the finetune of the pre-trained encoder, it is not required by our BadEncoder. It can be ignored if you do not need to monitor the finetune of the pre-trained encoder
            _ = test(model.f, memory_loader, test_loader_clean, test_loader_backdoor, epoch, args)
            # _ = test(model.f, memory_loader, test_loader_clean, test_loader_backdoor, epoch, args)

        elif args.encoder_usage_info == 'imagenet' or args.encoder_usage_info == 'CLIP':
            train_loss = train(model.visual, clean_model.visual, train_loader, optimizer, args,net)
            # train_loss = train(model.visual, clean_model.visual, train_loader, optimizer, args)

        else:
            raise NotImplementedError()

        # Save the BadEncoder
        if epoch % 50 == 0:
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict(),}, args.results_dir + '/model_' + str(epoch) + '.pth')


        now = datetime.now()
        print("当前时间：", now.strftime("%Y-%m-%d %H:%M:%S"))