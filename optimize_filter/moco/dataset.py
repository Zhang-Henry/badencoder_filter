import os, torch
from torch.utils import data
from PIL import Image
import numpy as np

class FileListDataset(data.Dataset):
    def __init__(self, path_to_txt_file, transform1, transform2):
        # self.data_root = data_root
        with open(path_to_txt_file, 'r') as f:
            self.file_list = f.readlines()
            self.file_list = [row.rstrip() for row in self.file_list]

        self.transform1 = transform1
        self.transform2 = transform2


    def __getitem__(self, idx):
        image_path = self.file_list[idx].split()[0]
        img = Image.open(image_path).convert('RGB')

        # img_trans1 = self.transform1(img)
        # img_trans2 = self.transform1(img)
        img= self.transform2(img)

        return img
        # return img_trans1,img_trans2, img # 第一个是经过各种变换以后的图片，第二个是只转换为tensor的图片

    def __len__(self):
        return len(self.file_list)
