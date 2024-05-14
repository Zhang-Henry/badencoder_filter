import sys
sys.path.append('..')

import os
import argparse
import random

import torchvision
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import get_encoder_architecture_usage
from datasets import get_shadow_dataset
from evaluation import test

from defenses.Beatrix.Beatrix import Feature_Correlations

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import matplotlib as mpl
from matplotlib.ticker import FuncFormatter
from util import filter_color_loss,clamp_batch_images
from optimize_filter.tiny_network import U_Net_tiny

import matplotlib
matplotlib.use('AGG')


def eval_beatrix(backdoored_encoder, clean_encoder, data_loader, train_optimizer, args,filter):
    backdoored_encoder.eval()

    train_bar = tqdm(data_loader)

    for module in backdoored_encoder.modules():
    # print(module)
        if isinstance(module, nn.BatchNorm2d):
            if hasattr(module, 'weight'):
                module.weight.requires_grad_(False)
            if hasattr(module, 'bias'):
                module.bias.requires_grad_(False)
            module.eval()

    clean_encoder.eval()

    for img_clean, img_backdoor_list, reference_list,reference_aug_list, _ in train_bar:
        img_clean = img_clean.cuda(non_blocking=True)
        img_backdoor_cuda_list = []

        for img_backdoor in img_backdoor_list:
            img_backdoor_cuda_list.append(img_backdoor.cuda(non_blocking=True))

        with torch.no_grad():
            clean_feature_raw = clean_encoder(img_clean)
            #clean_feature_raw = F.normalize(clean_feature_raw, dim=-1)

        feature_raw = backdoored_encoder(img_clean)
        #feature_raw = F.normalize(feature_raw, dim=-1)

        feature_backdoor_list = []
        for img_backdoor in img_backdoor_cuda_list:

            # img_backdoor=filter(img_backdoor)
            # img_backdoor= clamp_batch_images(img_backdoor,args)

            feature_backdoor = backdoored_encoder(img_backdoor)
            #feature_backdoor = F.normalize(feature_backdoor, dim=-1)

    #feature_raw = torch.load("./GH_beatrix_features/feature_raw.pt").cuda()
    #feature_backdoor = torch.load("./GH_beatrix_features/feature_backdoor.pt").cuda()

    ood_detection = Feature_Correlations(POWER_list=np.arange(1,9),mode='mad')
    ood_detection.train(in_data=[feature_raw])

    print(ood_detection.get_deviations_([feature_raw]))
    print(ood_detection.get_deviations_([feature_backdoor]))

    clean_deviations = ood_detection.get_deviations_([feature_raw])
    backdoor_deviations = ood_detection.get_deviations_([feature_backdoor])

    test_deviations = np.concatenate(clean_deviations,0)
    test_deviations_sort = np.sort(test_deviations,0)
    threshold_95 = test_deviations_sort[int(len(test_deviations_sort)*0.95)]
    threshold_99 = test_deviations_sort[int(len(test_deviations_sort)*0.99)]
    print(threshold_95)
    print(threshold_99)

    clean_label_test = np.zeros((clean_deviations.shape[0],))
    bd_label_test = np.ones((backdoor_deviations.shape[0],))

    test_deviations = np.concatenate((clean_deviations,backdoor_deviations),0)
    label_test = np.concatenate([clean_label_test,bd_label_test],0)

    ood_label_95 = np.where(test_deviations > threshold_95, 1, 0).squeeze()
    ood_label_99 = np.where(test_deviations > threshold_99, 1, 0).squeeze()

    false_negetive_95 = np.where(label_test - ood_label_95 > 0, 1, 0).squeeze()
    false_negetive_99 = np.where(label_test - ood_label_99 > 0, 1, 0).squeeze()
    false_positive_95 = np.where(label_test - ood_label_95 < 0, 1, 0).squeeze()
    false_positive_99 = np.where(label_test - ood_label_99 < 0, 1, 0).squeeze()

    print(f'false_negetive_95:{false_negetive_95.sum()},false_negetive_99:{false_negetive_99.sum()}')
    print(f'false_positive_95:{false_positive_95.sum()},false_positive_99：{false_positive_99.sum()}')

    print("total number:",test_deviations.shape[0])

    avg_score_diff_clean_list0 = ood_detection.get_deviations_([feature_raw]).squeeze()
    avg_score_diff_trojan_list0 = ood_detection.get_deviations_([feature_backdoor]).squeeze()

    cut_threshold = args.cut_threshold
    avg_score_diff_clean_list = []
    for score in avg_score_diff_clean_list0:
        if score< cut_threshold:
            avg_score_diff_clean_list.append(score)

    avg_score_diff_trojan_list = []
    for score in avg_score_diff_trojan_list0:
        if score< cut_threshold:
            avg_score_diff_trojan_list.append(score)

    print(len(avg_score_diff_clean_list))
    print(len(avg_score_diff_trojan_list))

    bins_clean = 28
    bins_trojan = 28
    plt.hist(avg_score_diff_clean_list, bins_clean, weights=np.ones(len(avg_score_diff_clean_list)) / len(avg_score_diff_clean_list), alpha=0.5, label='Clean')
    plt.hist(avg_score_diff_trojan_list, bins_trojan, weights=np.ones(len(avg_score_diff_trojan_list)) / len(avg_score_diff_trojan_list), alpha=0.5, label='Backdoor')
    #plt.hist(avg_score_diff_clean_list, bins,alpha=0.5, label='w/o Trojan')
    #plt.hist(avg_score_diff_trojan_list, bins,alpha=0.5, label='w/ Trojan')


    #plt.legend(loc='upper right', fontsize = 22)
    #plt.legend(loc='best', fontsize = 20)
    plt.legend(loc='best', fontsize = 21)
    plt.tick_params(labelsize=26)
    plt.xlabel('Deviation', fontsize = 26)
    plt.ylabel('Percentage', fontsize = 26)
    #plt.title('avg_score_diff', fontsize = 20)

    plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    #plt.xlim((0, 5))
    #plt.xlim((0, 1.95))
    plt.tight_layout()
    plt.savefig(args.save_name, dpi=600)
    plt.close('all')


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
    parser.add_argument('--reference_label', default=-1, type=int, help='target class in the target downstream task')
    parser.add_argument('--cut_threshold', default=200, type=float, help='target class in the target downstream task')
    parser.add_argument('--save_name', default='./beatrix.png', type=str, metavar='PATH', help='path to save the backdoored encoder')

    parser.add_argument('--results_dir', default='', type=str, metavar='PATH', help='path to save the backdoored encoder')
    parser.add_argument('--rand_init', action='store_true')

    parser.add_argument('--seed', default=100, type=int, help='which seed the code runs on')
    parser.add_argument('--gpu', default='0', type=str, help='which gpu the code runs on')
    args = parser.parse_args()

    # Set the seed and determine the GPU
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Specify the pre-training data directory
    args.data_dir = f'./data/{args.shadow_dataset.split("_")[0]}/'
    args.knn_k = 200
    args.knn_t = 0.5
    args.reference_label = 0
    print(args)

    args.shadow_fraction = 0.2

    args.pretraining_dataset = args.shadow_dataset

    # Create the Pytorch Datasets, and create the data loader for the training set
    # memory_data, test_data_clean, and test_data_backdoor are used to monitor the finetuning process. They are not reqruied by our BadEncoder
    shadow_data, memory_data, test_data_clean, test_data_backdoor = get_shadow_dataset(args)
    train_loader = DataLoader(shadow_data, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)

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
    #     test_acc_1 = test(model.f, memory_loader, test_loader_clean, test_loader_backdoor,0, args)
    #     print('initial test acc: {}'.format(test_acc_1))

    net = U_Net_tiny(img_ch=3,output_ch=3)
    # if not args.rand_init:
    #     state_dict = torch.load(args.trigger_file, map_location=torch.device('cuda:0'))
    #     net.load_state_dict(state_dict['model_state_dict'])
    # net=net.cuda().eval()

    eval_beatrix(model.f, clean_model.f, train_loader, optimizer, args, net)


    '''# training loop
    for epoch in range(1, args.epochs + 1):
        print("=================================================")
        if args.encoder_usage_info == 'cifar10' or args.encoder_usage_info == 'stl10':
            train_loss = train(model.f, clean_model.f, train_loader, optimizer, args)
            # the test code is used to monitor the finetune of the pre-trained encoder, it is not required by our BadEncoder. It can be ignored if you do not need to monitor the finetune of the pre-trained encoder
            _ = test(model.f, memory_loader, test_loader_clean, test_loader_backdoor,epoch, args)
        elif args.encoder_usage_info == 'imagenet' or args.encoder_usage_info == 'CLIP':
            train_loss = train(model.visual, clean_model.visual, train_loader, optimizer, args)
        else:
            raise NotImplementedError()

        # Save the BadEncoder
        if epoch % args.epochs == 0:
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict(),}, args.results_dir + '/model_' + str(epoch) + '.pth')

        # Save the intermediate checkpoint
        # torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict(),}, args.results_dir + '/model_last.pth')'''
