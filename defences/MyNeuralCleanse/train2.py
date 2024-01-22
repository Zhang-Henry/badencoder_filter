import sys
sys.path.append('../../')
sys.path.append('../../../')
import torch
import numpy as np
from torch.nn import CrossEntropyLoss
import tqdm
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from data import get_data

import os
import argparse

import torch.nn as nn
import torch.nn.functional as F

from datasets import get_dataset_evaluation
from models import get_encoder_architecture_usage
from evaluation import NeuralNet
from datetime import datetime


def train(model, target_label, train_loader,encoder):
    print("Processing label: {}".format(target_label))

    width, height = 32,32
    trigger = torch.rand((3, width, height), requires_grad=True)
    trigger = trigger.to(device).detach().requires_grad_(True)
    mask = torch.rand((width, height), requires_grad=True)
    mask = mask.to(device).detach().requires_grad_(True)


    min_norm = np.inf
    min_norm_count = 0

    criterion = CrossEntropyLoss()
    optimizer = torch.optim.Adam([{"params": trigger},{"params": mask}],lr=0.005)
    model.to(device)
    model.eval()

    for epoch in range(args.nn_epochs):
        norm = 0.0
        for images, _ in tqdm.tqdm(train_loader, desc='Epoch %3d' % (epoch + 1)):
            optimizer.zero_grad()
            images = images.to(device)
            trojan_images = (1 - torch.unsqueeze(mask, dim=0)) * images + torch.unsqueeze(mask, dim=0) * trigger
            feature_trojan = encoder(trojan_images)
            feature_trojan = F.normalize(feature_trojan, dim=1)

            output = model(feature_trojan)
            y_pred = output.argmax(dim=1)

            y_target = torch.full((y_pred.size(0),), target_label, dtype=torch.long).to(device)
            # print('y_pred',y_pred)
            # print('y_target',y_target)
            loss = criterion(y_pred.float(), y_target.float()) + args.lamda * torch.sum(torch.abs(mask))
            loss.backward()
            optimizer.step()

            # figure norm
            with torch.no_grad():
                # 防止trigger和norm越界
                torch.clip_(trigger, 0, 1)
                torch.clip_(mask, 0, 1)
                norm = torch.sum(torch.abs(mask))
        print("norm: {}".format(norm))

        # to early stop
        if norm < min_norm:
            min_norm = norm
            min_norm_count = 0
        else:
            min_norm_count += 1

        if min_norm_count > 30:
            break

    return trigger.cpu(), mask.cpu()


def reverse_engineer(args,train_loader,encoder,classifier):

    norm_list = []
    for label in range(args.num_classes):
        trigger, mask = train(classifier, label, train_loader,encoder)
        norm_list.append(mask.sum().item())

        trigger = trigger.cpu().detach().numpy()
        trigger = np.transpose(trigger, (1,2,0))
        plt.axis("off")
        plt.imshow(trigger)
        plt.savefig(f'mask/{args.dataset}/trigger_{label}.png', bbox_inches='tight', pad_inches=0.0)

        mask = mask.cpu().detach().numpy()
        plt.axis("off")
        plt.imshow(mask)
        plt.savefig(f'mask/{args.dataset}/mask_{label}.png', bbox_inches='tight', pad_inches=0.0)

    print(norm_list)



if __name__ == "__main__":

    now = datetime.now()
    print("当前时间：", now.strftime("%Y-%m-%d %H:%M:%S"))

    parser = argparse.ArgumentParser(description='Evaluate the clean or backdoored encoders')
    parser.add_argument('--dataset', default='cifar10', type=str, help='downstream dataset')
    parser.add_argument('--encoder_usage_info', default='', type=str, help='used to locate encoder usage info, e.g., encoder architecture and input normalization parameter')
    parser.add_argument('--encoder', default='', type=str, help='path to the image encoder')
    parser.add_argument('--classifier', default='', type=str, help='path to the image encoder')
    parser.add_argument('--gpu', default='0', type=str, help='the index of gpu used to train the model')
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--lamda', default=0.01, type=float)
    parser.add_argument('--nn_epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=64, type=int, metavar='N', help='mini-batch size')
    parser.add_argument('--num_classes', default=10, type=int)
    parser.add_argument('--noise', type=str)
    parser.add_argument('--reference_file', default='/home/hrzhang/projects/badencoder_filter/reference/cifar10/one.npz', type=str, help='path to the reference file (default: none)')
    parser.add_argument('--trigger_file', default='', type=str)
    parser.add_argument('--reference_label', default=0, type=str)

    args = parser.parse_args()


    args.data_dir = f'../../data/{args.dataset}/'
    target_dataset, train_data, test_data_clean, test_data_backdoor = get_dataset_evaluation(args)

    test_loader_clean = DataLoader(test_data_clean, batch_size=args.batch_size, shuffle=False, num_workers=8,
                                   pin_memory=True)

    args.num_classes = len(train_data.classes)

    encoder = get_encoder_architecture_usage(args).cuda()

    if args.encoder != '':
        print('Loaded from: {}'.format(args.encoder))
        checkpoint = torch.load(args.encoder)
        args_v = checkpoint.get('args', None)
        loss = checkpoint.get('loss', None)
        if args_v:
            print(args_v)
        if loss:
            print(loss)

        encoder.load_state_dict(checkpoint['state_dict'])


    classifier = NeuralNet(512, [512, 256], args.num_classes).cuda()
    classifier.load_state_dict(torch.load(args.classifier))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    reverse_engineer(args,test_loader_clean,encoder.f,classifier)









