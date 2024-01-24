import torch
import os
import torchvision
import numpy as np
import cv2
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from config import get_argument

import sys
sys.path.append('../../')
sys.path.append('../../../')


import torch.nn as nn
import torch.nn.functional as F

from datasets import get_dataset_evaluation
from models import get_encoder_architecture_usage
from evaluation import NeuralNet
from datetime import datetime
from util import clamp_batch_images
from optimize_filter.tiny_network import U_Net_tiny
from datasets.backdoor_dataset import CIFAR10Mem


class ToNumpy:
    def __call__(self, x):
        x = np.array(x)
        if len(x.shape) == 2:
            x = np.expand_dims(x, axis=2)
        return x

class Denormalizer:
    def __init__(self, opt):
        self.denormalizer = self._get_denormalizer(opt)

    def _get_denormalizer(self, opt):
        if opt.encoder_usage_info == "cifar10":
            denormalizer = Denormalize(opt, [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        elif opt.encoder_usage_info == "stl10":
            denormalizer = Denormalize(opt, [0.44087798, 0.42790666, 0.38678814], [0.25507198, 0.24801506, 0.25641308])
        return denormalizer

    def __call__(self, x):
        if self.denormalizer:
            x = self.denormalizer(x)
        return x

class Normalize:
    def __init__(self, opt, expected_values, variance):
        self.n_channels = opt.input_channel
        self.expected_values = expected_values
        self.variance = variance
        assert self.n_channels == len(self.expected_values)

    def __call__(self, x):
        x_clone = x.clone()
        for channel in range(self.n_channels):
            x_clone[:, :, channel] = (x[:, :, channel] - self.expected_values[channel]) / self.variance[channel]
        return x_clone


class Denormalize:
    def __init__(self, opt, expected_values, variance):
        self.n_channels = opt.input_channel
        self.expected_values = expected_values
        self.variance = variance
        assert self.n_channels == len(self.expected_values)

    def __call__(self, x):
        x_clone = x.clone()
        for channel in range(self.n_channels):
            x_clone[:, :, channel] = x[:, :, channel] * self.variance[channel] + self.expected_values[channel]
        return x_clone


class STRIP:
    # def _superimpose(self, background, overlay):
    #     output = cv2.addWeighted(background, 1, overlay.cpu().numpy(), 1, 0)
    #     if len(output.shape) == 2:
    #         output = np.expand_dims(output, 2)
    #     return output

    def _superimpose(self, background, overlay):
        output = cv2.addWeighted(background, 1, overlay, 1, 0)
        if len(output.shape) == 2:
            output = np.expand_dims(output, 2)
        return output

    def _get_entropy(self, background, dataset, classifier, encoder):
        entropy_sum = [0] * self.n_sample
        x1_add = [0] * self.n_sample
        index_overlay = np.random.randint(0, len(dataset), size=self.n_sample)
        for index in range(self.n_sample):
            add_image = self._superimpose(background, dataset[index_overlay[index]][0])
            add_image = self.normalize(add_image)
            # add_image = torch.from_numpy(add_image)
            x1_add[index] = add_image

        data = torch.stack(x1_add).to(self.device)
        feature = encoder(data)
        feature = F.normalize(feature[0], dim=1)
        py1_add_ = classifier(feature)
        py1_add = torch.sigmoid(py1_add_).detach().cpu().numpy()
        # py1_add = torch.relu(py1_add_).detach().cpu().numpy()
        # py1_add = py1_add.detach().cpu().numpy()
        entropy_sum = -np.nansum(py1_add * np.log2(py1_add))
        return entropy_sum / self.n_sample

    def _get_denormalize(self, opt):
        if opt.encoder_usage_info == "cifar10":
            denormalizer = Denormalize(opt, [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        elif opt.encoder_usage_info == "stl10":
            denormalizer = Denormalize(opt, [0.44087798, 0.42790666, 0.38678814], [0.25507198, 0.24801506, 0.25641308])
        else:
            raise Exception("Invalid dataset")
        return denormalizer

    def _get_normalize(self, opt):
        if opt.encoder_usage_info == "cifar10":
            normalizer = Normalize(opt, [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        elif opt.encoder_usage_info == "stl10":
            normalizer = Normalize(opt, [0.44087798, 0.42790666, 0.38678814], [0.25507198, 0.24801506, 0.25641308])
        else:
            raise Exception("Invalid dataset")
        if normalizer:
            transform = transforms.Compose([transforms.ToTensor(), normalizer])
        else:
            transform = transforms.ToTensor()
        return transform

    def __init__(self, opt):
        super().__init__()
        self.n_sample = opt.n_sample
        self.normalizer = self._get_normalize(opt)
        self.denormalizer = self._get_denormalize(opt)
        self.device = opt.device

    def normalize(self, x):
        if self.normalizer:
            x = self.normalizer(x)
        return x

    def denormalize(self, x):
        if self.denormalizer:
            x = self.denormalizer(x)
        return x

    def __call__(self, background, dataset, classifier, encoder):
        return self._get_entropy(background, dataset, classifier, encoder)


def create_backdoor(inputs, opt):
    state_dict = torch.load(opt.trigger_file, map_location=torch.device('cuda:0'))
    filter = U_Net_tiny(img_ch=3,output_ch=3)
    filter.load_state_dict(state_dict['model_state_dict'])
    filter=filter.cuda().eval()
    bd_inputs = filter(inputs)
    bd_inputs = clamp_batch_images(bd_inputs,opt)
    return bd_inputs



def strip(opt, mode="clean"):
    # Load pretrained model
    mode = opt.attack_mode
    opt.ckpt_folder = os.path.join(opt.checkpoints, opt.dataset)
    opt.ckpt_path = os.path.join(opt.ckpt_folder, "{}_{}_morph.pth.tar".format(opt.dataset, mode))
    opt.log_dir = os.path.join(opt.ckpt_folder, "log_dir")


    encoder = get_encoder_architecture_usage(opt).cuda()
    if opt.encoder != '':
        print('Loaded from: {}'.format(opt.encoder))
        checkpoint = torch.load(opt.encoder)
        # args_v = checkpoint.get('args', None)
        # loss = checkpoint.get('loss', None)
        # if args_v:
        #     print(args_v)
        # if loss:
        #     print(loss)

        encoder.load_state_dict(checkpoint['state_dict'])

    classifier = NeuralNet(512, [512, 256], opt.num_classes).cuda()
    classifier.load_state_dict(torch.load(opt.classifier))
    netC=classifier

    # Prepare test set
    opt.bs = opt.n_test

    opt.data_dir = f'../../data/{opt.dataset}/'
    target_dataset, train_data, test_data_clean, test_data_backdoor = get_dataset_evaluation(opt)
    test_dataloader = DataLoader(test_data_clean, batch_size=opt.batch_size, shuffle=False, num_workers=8,
                                   pin_memory=True)
    testset = CIFAR10Mem(numpy_file=opt.data_dir+'train.npz', class_type=[], transform=ToNumpy())
    denormalizer = Denormalizer(opt)

    # STRIP detector
    strip_detector = STRIP(opt)

    # Entropy list
    list_entropy_trojan = []
    list_entropy_benign = []

    if mode == "attack":
        # Testing with perturbed data
        print("Testing with bd data !!!!")
        inputs, targets = next(iter(test_dataloader))
        inputs = inputs.to(opt.device)


        bd_inputs = create_backdoor(inputs, opt)
        bd_inputs = denormalizer(bd_inputs) * 255.0
        bd_inputs = bd_inputs.detach().cpu().numpy()
        bd_inputs = np.clip(bd_inputs, 0, 255).astype(np.uint8).transpose((0, 2, 3, 1))

        for index in range(opt.n_test):
            background = bd_inputs[index]
            entropy = strip_detector(background, testset, netC, encoder)
            list_entropy_trojan.append(entropy)

        # Testing with clean data
        for index in range(opt.n_test):
            background, _ = testset[index]
            entropy = strip_detector(background, testset, netC, encoder)
            list_entropy_benign.append(entropy)
    else:
        # Testing with clean data
        print("Testing with clean data !!!!")
        for index in range(opt.n_test):
            background, _ = testset[index]
            entropy = strip_detector(background, testset, netC, encoder)
            list_entropy_benign.append(entropy)

    return list_entropy_trojan, list_entropy_benign


def main():
    opt = get_argument().parse_args()

    if opt.dataset == "cifar10" or opt.dataset == "svhn" or opt.dataset == 'stl10':
        opt.num_classes = 10
    elif opt.dataset == "gtsrb":
        opt.num_classes = 43
    opt.input_height = 32
    opt.input_width = 32
    opt.input_channel = 3


    if "2" in opt.attack_mode:
        mode = "attack"
    else:
        mode = "clean"

    lists_entropy_trojan = []
    lists_entropy_benign = []
    for test_round in range(opt.test_rounds):
        list_entropy_trojan, list_entropy_benign = strip(opt, mode)
        lists_entropy_trojan += list_entropy_trojan
        lists_entropy_benign += list_entropy_benign

    # Save result to file
    result_dir = os.path.join(opt.results, opt.dataset)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    result_path = os.path.join(result_dir, opt.attack_mode)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    result_path_trojan = os.path.join(result_path,"{}_{}_output_trojan.txt".format(opt.dataset, opt.attack_mode))
    result_path_clean = os.path.join(result_path,"{}_{}_output_clean.txt".format(opt.dataset, opt.attack_mode))

    with open(result_path_trojan, "w+") as f:
        for index in range(len(lists_entropy_trojan)):
            if index < len(lists_entropy_trojan) - 1:
                f.write("{} ".format(lists_entropy_trojan[index]))
            else:
                f.write("{}".format(lists_entropy_trojan[index]))

        f.write("\n")

    with open(result_path_clean, "w+") as f:
        for index in range(len(lists_entropy_benign)):
            if index < len(lists_entropy_benign) - 1:
                f.write("{} ".format(lists_entropy_benign[index]))
            else:
                f.write("{}".format(lists_entropy_benign[index]))

    min_entropy = min(lists_entropy_trojan + lists_entropy_benign)

    # Determining
    print("Min entropy trojan: {}, Detection boundary: {}".format(min_entropy, opt.detection_boundary))
    if min_entropy < opt.detection_boundary:
        print("A backdoored model\n")
    else:
        print("Not a backdoor model\n")


if __name__ == "__main__":
    main()
