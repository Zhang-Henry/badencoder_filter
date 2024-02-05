
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from optimize_filter.network import AttU_Net,U_Net
from optimize_filter.tiny_network import U_Net_tiny
from util import clamp_batch_images
from badencoder_bpp import back_to_np_4d,np_4d_to_tensor,floydDitherspeed

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size_list, num_classes):
        super(NeuralNet, self).__init__()
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(input_size, hidden_size_list[0])
        self.fc2 = nn.Linear(hidden_size_list[0], hidden_size_list[1])
        self.fc3 = nn.Linear(hidden_size_list[1], num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.dropout2(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        return out


def create_torch_dataloader(feature_bank, label_bank, batch_size, shuffle=False, num_workers=2, pin_memory=True):
    # transform to torch tensor
    tensor_x, tensor_y = torch.Tensor(feature_bank), torch.Tensor(label_bank)

    dataloader = DataLoader(
        TensorDataset(tensor_x, tensor_y),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return dataloader


def net_train(net, train_loader, optimizer, epoch, criterion):
    """Training"""
    net.train()
    overall_loss = 0.0
    for batch_idx, (data, label) in enumerate(train_loader):
        data, label = data.cuda(non_blocking=True), label.cuda(non_blocking=True)
        optimizer.zero_grad()
        output = net(data)
        loss = criterion(output, label.long())

        loss.backward()
        optimizer.step()
        overall_loss += loss.item()
    print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, overall_loss*train_loader.batch_size/len(train_loader.dataset)))


def net_test(net, test_loader, epoch, criterion, keyword='Accuracy'):
    """Testing"""
    net.eval()
    test_loss = 0.0
    correct = 0.0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            output = net(data)
            test_loss += criterion(output, target.long()).item()
            pred = output.argmax(dim=1, keepdim=True)

            correct += pred.eq(target.view_as(pred)).sum().item()

    test_acc = 100. * correct / len(test_loader.dataset)
    test_loss /= len(test_loader.dataset)
    print('{{"metric": "Eval - {}", "value": {}, "epoch": {}}}'.format(
        keyword, 100. * correct / len(test_loader.dataset), epoch))

    return test_acc


def predict_feature(args,net, data_loader,keyword='clean'):
    net.eval()
    feature_bank, target_bank = [], []
    # if keyword=='backdoor':
    #     state_dict = torch.load(args.trigger_file, map_location=torch.device('cuda:0'))
    #     filter = U_Net_tiny(img_ch=3,output_ch=3)
    #     filter.load_state_dict(state_dict['model_state_dict'])
    #     filter=filter.cuda().eval()

    with torch.no_grad():
        # generate feature bank
        for data, target in tqdm(data_loader, desc='Feature extracting'):
            data=data.cuda(non_blocking=True)
            ########## ours unet ########
            # if keyword=='backdoor':
            #     #########
            #     data=filter(data)
            #     data= clamp_batch_images(data,args)

            ########## wanet #######
            # if keyword=='backdoor':
            #     bs=data.shape[0]
            #     input_height=32
            #     grid_rescale=1
            #     s=0.5
            #     k=4
            #     ins = torch.rand(1, 2, k, k) * 2 - 1
            #     ins = ins / torch.mean(torch.abs(ins))
            #     noise_grid = (
            #         F.upsample(ins, size=input_height, mode="bicubic", align_corners=True)
            #         .permute(0, 2, 3, 1)
            #         .cuda()
            #     )
            #     array1d = torch.linspace(-1, 1, steps=input_height)
            #     x, y = torch.meshgrid(array1d, array1d)
            #     identity_grid = torch.stack((y, x), 2)[None, ...].cuda()
            #     grid_temps = (identity_grid + s * noise_grid / input_height) * grid_rescale
            #     grid_temps = torch.clamp(grid_temps, -1, 1)

            #     data = F.grid_sample(data, grid_temps.repeat(bs, 1, 1, 1), align_corners=True)
                ########## bpp ##########
            # if keyword=='backdoor':
            #     inputs_bd = back_to_np_4d(data,args)

            #     for i in range(inputs_bd.shape[0]):
            #         inputs_bd[i,:,:,:] = torch.round(torch.from_numpy(floydDitherspeed(inputs_bd[i].detach().cpu().numpy(),float(8))).cuda())

            #     data = np_4d_to_tensor(inputs_bd,args)
            ########################################
            feature = net(data)
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
            target_bank.append(target)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).contiguous()
        target_bank = torch.cat(target_bank, dim=0).contiguous()

    return feature_bank.cpu().detach().numpy(), target_bank.detach().numpy()
