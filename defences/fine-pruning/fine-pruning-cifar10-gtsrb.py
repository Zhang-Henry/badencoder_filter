import torch
import os
import torch.nn as nn
import copy
import torch.nn.functional as F
from config import get_arguments


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
from utils import progress_bar
from torch.utils.data import DataLoader

def create_targets_bd(targets, opt):
    if opt.attack_mode == "all2one":
        bd_targets = torch.ones_like(targets) * opt.target_label
    elif opt.attack_mode == "all2all":
        bd_targets = torch.tensor([(label + 1) % opt.num_classes for label in targets])
    else:
        raise Exception("{} attack mode is not implemented".format(opt.attack_mode))
    return bd_targets.to(opt.device)


def create_bd(netG, netM, inputs, targets, opt):
    bd_targets = create_targets_bd(targets, opt)
    patterns = netG(inputs)
    patterns = netG.normalize_pattern(patterns)

    masks_output = netM.threshold(netM(inputs))
    bd_inputs = inputs + (patterns - inputs) * masks_output
    return bd_inputs, bd_targets


def eval(netC, identity_grid, noise_grid, test_dl, opt):
    print(" Eval:")
    acc_clean = 0.0
    acc_bd = 0.0
    total_sample = 0
    total_correct_clean = 0
    total_correct_bd = 0

    for batch_idx, (inputs, targets) in enumerate(test_dl):
        inputs, targets = inputs.to(opt.device), targets.to(opt.device)
        bs = inputs.shape[0]
        total_sample += bs

        # Evaluating clean
        preds_clean = netC(inputs)
        correct_clean = torch.sum(torch.argmax(preds_clean, 1) == targets)
        total_correct_clean += correct_clean
        acc_clean = total_correct_clean * 100.0 / total_sample

        # Evaluating backdoor
        grid_temps = (identity_grid + opt.s * noise_grid / opt.input_height) * opt.grid_rescale
        grid_temps = torch.clamp(grid_temps, -1, 1)

        inputs_bd = F.grid_sample(inputs, grid_temps.repeat(bs, 1, 1, 1), align_corners=True)
        targets_bd = torch.ones_like(targets) * opt.target_label
        preds_bd = netC(inputs_bd)
        correct_bd = torch.sum(torch.argmax(preds_bd, 1) == targets_bd)
        total_correct_bd += correct_bd
        acc_bd = total_correct_bd * 100.0 / total_sample

        progress_bar(batch_idx, len(test_dl), "Acc Clean: {:.3f} | Acc Bd: {:.3f}".format(acc_clean, acc_bd))
    return acc_clean, acc_bd


def main():
    # Prepare arguments
    opt = get_arguments().parse_args()
    if opt.dataset == "cifar10" or opt.dataset == "svhn" or opt.dataset == 'stl10':
        opt.num_classes = 10
    elif opt.dataset == "gtsrb":
        opt.num_classes = 43
    opt.input_height = 32
    opt.input_width = 32
    opt.input_channel = 3

    encoder = get_encoder_architecture_usage(opt).cuda()
    print('Loaded from: {}'.format(opt.encoder))
    checkpoint = torch.load(opt.encoder)
    encoder.load_state_dict(checkpoint['state_dict'])
    encoder.eval()
    encoder.requires_grad_(False)


    classifier = NeuralNet(512, [512, 256], opt.num_classes).cuda()
    classifier.load_state_dict(torch.load(opt.classifier))
    classifier.eval()
    classifier.requires_grad_(False)

    netC=classifier


    opt.data_dir = f'../../data/{opt.dataset}/'
    target_dataset, train_data, test_data_clean, test_data_backdoor = get_dataset_evaluation(opt)
    opt.ckpt_folder = os.path.join(opt.checkpoints, opt.dataset)
    opt.log_dir = os.path.join(opt.ckpt_folder, "log_dir")


    test_dl = DataLoader(test_data_clean, batch_size=opt.batch_size, shuffle=False, num_workers=8,
                                   pin_memory=True)
    # Forward hook for getting layer's output
    container = []

    def forward_hook(module, input, output):
        container.append(output)

    hook = netC.fc3.register_forward_hook(forward_hook)

    # Forwarding all the validation set
    print("Forwarding all the validation dataset:")
    for batch_idx, (inputs, _) in enumerate(test_dl):
        inputs = inputs.to(opt.device)
        netC(inputs)
        progress_bar(batch_idx, len(test_dl))

    # Processing to get the "more important mask"
    container = torch.cat(container, dim=0)
    activation = torch.mean(container, dim=[0, 2, 3])
    seq_sort = torch.argsort(activation)
    pruning_mask = torch.ones(seq_sort.shape[0], dtype=bool)
    hook.remove()

    # Pruning times - no-tuning after pruning a channel!!!
    acc_clean = []
    acc_bd = []
    opt.outfile = "{}_results.txt".format(opt.dataset)
    with open(opt.outfile, "w") as outs:
        for index in range(pruning_mask.shape[0]):
            net_pruned = copy.deepcopy(netC)
            num_pruned = index
            if index:
                channel = seq_sort[index - 1]
                pruning_mask[channel] = False
            print("Pruned {} filters".format(num_pruned))

            net_pruned.layer4[1].conv2 = nn.Conv2d(
                pruning_mask.shape[0], pruning_mask.shape[0] - num_pruned, (3, 3), stride=1, padding=1, bias=False
            )
            net_pruned.linear = nn.Linear(pruning_mask.shape[0] - num_pruned, 10)

            # Re-assigning weight to the pruned net
            for name, module in net_pruned._modules.items():
                if "layer4" in name:
                    module[1].conv2.weight.data = netC.layer4[1].conv2.weight.data[pruning_mask]
                    module[1].ind = pruning_mask
                elif "linear" == name:
                    module.weight.data = netC.linear.weight.data[:, pruning_mask]
                    module.bias.data = netC.linear.bias.data
                else:
                    continue
            net_pruned.to(opt.device)
            clean, bd = eval(net_pruned, identity_grid, noise_grid, test_dl, opt)
            outs.write("%d %0.4f %0.4f\n" % (index, clean, bd))


if __name__ == "__main__":
    main()
