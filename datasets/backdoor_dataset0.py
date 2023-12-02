
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from PIL import Image
import numpy as np
import torch
import random
import pilgram
import torch.nn.functional as F

import copy
# from .CTRL.utils.frequency import PoisonFre



class ReferenceImg(Dataset):

    def __init__(self, reference_file, transform=None):
        """
        Args:
            numpy_file (string): Path to the numpy file.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.target_input_array = np.load(reference_file)

        self.data = self.target_input_array['x']
        self.targets = self.target_input_array['y']

        self.transform = transform

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)


class BadEncoderDataset(Dataset):

    def __init__(self, numpy_file, trigger_file, reference_file, indices, class_type, transform=None, bd_transform=None, ftt_transform=None):
        self.input_array = np.load(numpy_file)
        self.data = self.input_array['x']

        # self.trigger_input_array = np.load(trigger_file)
        self.target_input_array = np.load(reference_file)

        # self.trigger_patch_list = self.trigger_input_array['t']
        # self.trigger_mask_list = self.trigger_input_array['tm']
        self.target_image_list = self.target_input_array['x']

        self.classes = class_type
        self.indices = indices
        self.transform = transform
        self.bd_transform = bd_transform
        self.ftt_transform = ftt_transform
        self.filter = torch.load('trigger/filter.pt', map_location=torch.device('cpu'))

    def __getitem__(self, index):
        img = self.data[self.indices[index]]
        img_copy = copy.deepcopy(img)
        backdoored_image = copy.deepcopy(img)
        img = Image.fromarray(img)
        '''original image'''
        if self.transform is not None:
            im_1 = self.transform(img)
        img_raw = self.bd_transform(img)
        '''generate backdoor image'''

        img_backdoor_list = []
        for i in range(len(self.target_image_list)):

            ###########################
            ### for ins filter only ###

            # image_pil = Image.fromarray(img_copy)
            # filtered_image_pil = pilgram.kelvin(image_pil)  # 使用 _1977, xpro2, kelvin 滤镜

            # backdoored_image = np.array(filtered_image_pil)  # 将 PIL Image 对象转换回 NumPy 数组
            # img_backdoor =self.bd_transform(Image.fromarray(backdoored_image))

            ###########################
            ### origin ###

            # backdoored_image[:,:,:] = img_copy * self.trigger_mask_list[i] + self.trigger_patch_list[i][:]
            # img_backdoor =self.bd_transform(Image.fromarray(backdoored_image))

            ###########################
            # for customized filter only

            img_copy=torch.Tensor(img_copy)
            backdoored_image = F.conv2d(img_copy.permute(2, 0, 1), self.filter, padding=7//2)
            img_backdoor = self.bd_transform(backdoored_image.permute(1,2,0).detach().numpy())

            ###########################
            # for ctrl only
            # trans=transforms.Compose([
            #         transforms.ToTensor()
            #     ])

            # image_pil = Image.fromarray(img_copy)
            # tensor_image = trans(image_pil)

            # base_image=tensor_image.unsqueeze(0)
            # poison_frequency_agent = PoisonFre('args',32, [1,2], 32, [15,31],  False,  True)

            # x_tensor,_ = poison_frequency_agent.Poison_Frequency_Diff(base_image,0, 100.0)
            # img_backdoor = x_tensor.squeeze()
            # img_backdoor = np.clip(img_backdoor, 0, 1) #限制颜色范围在0-1

            # img_backdoor = self.bd_transform(img_backdoor.detach().numpy())

            ###########################
            img_backdoor_list.append(img_backdoor)


        target_image_list_return, target_img_1_list_return = [], []
        for i in range(len(self.target_image_list)):
            target_img = Image.fromarray(self.target_image_list[i])
            target_image = self.bd_transform(target_img)
            target_img_1 = self.ftt_transform(target_img)
            target_image_list_return.append(target_image)
            target_img_1_list_return.append(target_img_1)

        return img_raw, img_backdoor_list, target_image_list_return, target_img_1_list_return

    def __len__(self):
        return len(self.indices)


class BadEncoderTestBackdoor(Dataset):

    def __init__(self, numpy_file, trigger_file, reference_label, transform=None):
        """
        Args:
            numpy_file (string): Path to the numpy file.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.input_array = np.load(numpy_file)
        self.data = self.input_array['x']
        self.targets = self.input_array['y']


        # self.trigger_input_array = np.load(trigger_file)

        # self.trigger_patch_list = self.trigger_input_array['t']
        # self.trigger_mask_list = self.trigger_input_array['tm']

        self.target_class = reference_label

        self.test_transform = transform

        self.filter = torch.load('trigger/filter.pt', map_location=torch.device('cpu'))

    def __getitem__(self,index):
        img = copy.deepcopy(self.data[index])

        ###########################
        ### for ins filter only ###

        # image_pil = Image.fromarray(img)
        # filtered_image_pil = pilgram.kelvin(image_pil)
        # img_backdoor =self.test_transform(filtered_image_pil)

        ###########################

        # img[:] =img * self.trigger_mask_list[0] + self.trigger_patch_list[0][:]
        # img_backdoor =self.test_transform(Image.fromarray(img))

        ###########################
        # for customized filter only

        img_copy=torch.Tensor(img)
        backdoored_image = F.conv2d(img_copy.permute(2, 0, 1), self.filter, padding=7//2)
        img_backdoor = self.test_transform(backdoored_image.permute(1,2,0).detach().numpy())

        ###########################
        ###########################
        # for ctrl only
        # trans=transforms.Compose([
        #         transforms.ToTensor()
        #     ])

        # image_pil = Image.fromarray(img)
        # tensor_image = trans(image_pil)

        # base_image=tensor_image.unsqueeze(0)
        # poison_frequency_agent = PoisonFre('args',32, [1,2], 32, [15,31],  False,  True)

        # x_tensor,_ = poison_frequency_agent.Poison_Frequency_Diff(base_image,0, 100.0)
        # img_backdoor = x_tensor.squeeze()
        # img_backdoor = np.clip(img_backdoor, 0, 1) #限制颜色范围在0-1

        # img_backdoor = self.bd_transform(img_backdoor.detach().numpy())

        ###########################
        return img_backdoor, self.target_class


    def __len__(self):
        return self.data.shape[0]



class CIFAR10CUSTOM(Dataset):

    def __init__(self, numpy_file, class_type, transform=None):
        """
        Args:
            numpy_file (string): Path to the numpy file.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.input_array = np.load(numpy_file)
        self.data = self.input_array['x']
        self.targets = self.input_array['y'][:,0].tolist()
        self.classes = class_type
        self.transform = transform
    def __len__(self):
        return self.data.shape[0]


class CIFAR10Pair(CIFAR10CUSTOM):
    """CIFAR10 Dataset.
    """
    def __getitem__(self, index):
        img = self.data[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            im_1 = self.transform(img)
            im_2 = self.transform(img)

        return im_1, im_2


class CIFAR10Mem(CIFAR10CUSTOM):
    """CIFAR10 Dataset.
    """
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        return img, target


class CIFAR10M(CIFAR10CUSTOM):
    """CIFAR10 Dataset.
    """
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img_trans = self.transform(img)

        return img, img_trans
