from torchvision.datasets.vision import VisionDataset

from PIL import Image

import os
import os.path
from typing import Any, Callable, cast, Dict, List, Optional, Tuple

import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from PIL import Image
import numpy as np
import torch
import random
import torch.nn.functional as F
import torch.nn as nn
from optimize_filter.network import AttU_Net
from optimize_filter.tiny_network import U_Net_tiny

import copy

# import bchlib

# import tensorflow as tf
# from tensorflow.python.saved_model import tag_constants
# from tensorflow.python.saved_model import signature_constants
# from .CTRL.utils.frequency import PoisonFre


def has_file_allowed_extension(filename: str, extensions: Tuple[str, ...]) -> bool:
    # Checks if a file is an allowed extension.
    # Args:
    #     filename (string): path to a file
    #     extensions (tuple of strings): extensions to consider (lowercase)
    # Returns:
    #     bool: True if the filename ends with one of given extensions
    return filename.lower().endswith(extensions)


def is_image_file(filename: str) -> bool:
    """Checks if a file is an allowed image extension.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def make_dataset(
    directory: str,
    class_to_idx: Dict[str, int],
    extensions: Optional[Tuple[str, ...]] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None,
) -> List[Tuple[str, int]]:
    """Generates a list of samples of a form (path_to_sample, class).
    Args:
        directory (str): root dataset directory
        class_to_idx (Dict[str, int]): dictionary mapping class name to class index
        extensions (optional): A list of allowed extensions.
            Either extensions or is_valid_file should be passed. Defaults to None.
        is_valid_file (optional): A function that takes path of a file
            and checks if the file is a valid file
            (used to check of corrupt files) both extensions and
            is_valid_file should not be passed. Defaults to None.
    Raises:
        ValueError: In case ``extensions`` and ``is_valid_file`` are None or both are not None.
    Returns:
        List[Tuple[str, int]]: samples of a form (path_to_sample, class)
    """
    instances = []
    directory = os.path.expanduser(directory)
    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(x, cast(Tuple[str, ...], extensions))
    is_valid_file = cast(Callable[[str], bool], is_valid_file)
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = path, class_index
                    instances.append(item)
    return instances


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

# TODO: specify the return type
def accimage_loader(path: str) -> Any:
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def default_loader(path: str) -> Any:
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


is_valid_file = None

class BadEncoderDataset(VisionDataset):
    """A generic data loader where the samples are arranged in this way: ::
        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/[...]/xxz.ext
        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/[...]/asd932_.ext
    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.
     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(
            self,
            root: str,
            trigger_file, reference_file, indices, class_type,
            loader: Callable[[str], Any] = default_loader,
            transform=None, bd_transform=None, ftt_transform=None,
            extensions: Optional[Tuple[str, ...]] = IMG_EXTENSIONS if is_valid_file is None else None,
            target_transform: Optional[Callable] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> None:
        super(BadEncoderDataset, self).__init__(root, transform=transform,
                                            target_transform=target_transform)
        classes, class_to_idx = self._find_classes(self.root)
        samples = self.make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            if extensions is not None:
                msg += "Supported extensions are: {}".format(",".join(extensions))
            raise RuntimeError(msg)

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        #self.targets = [s[1] for s in samples]
        self.target_input_array = np.load(reference_file)

        # self.trigger_input_array = np.load(trigger_file)
        # self.trigger_patch_list = self.trigger_input_array['t']
        # self.trigger_mask_list = self.trigger_input_array['tm']

        #carlini
        # if not isinstance(self.trigger_patch_list, list):
        #     self.trigger_patch_list = [self.trigger_patch_list]
        #     self.trigger_mask_list = [self.trigger_mask_list]

        self.target_image_list = self.target_input_array['x']

        self.classes = class_type
        self.indices = indices
        self.transform = transform
        self.bd_transform = bd_transform
        self.ftt_transform = ftt_transform

        # self.filter = torch.load('trigger/filter.pt', map_location=torch.device('cpu'))

        # state_dict = torch.load(trigger_file, map_location=torch.device('cpu'))
        # self.net = U_Net_tiny(img_ch=3,output_ch=3)
        # self.net.load_state_dict(state_dict['model_state_dict'])
        # self.net=self.net.eval()



    @staticmethod
    def make_dataset(
        directory: str,
        class_to_idx: Dict[str, int],
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:
        return make_dataset(directory, class_to_idx, extensions=extensions, is_valid_file=is_valid_file)

    def _find_classes(self, dir: str) -> Tuple[List[str], Dict[str, int]]:
        """
        Finds the class folders in a dataset.
        Args:
            dir (string): Root directory path.
        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[self.indices[index]]
        sample = self.loader(path)
        # sample.save("/data/local/wzt/model_fix/BadEncoder/imagenet_filter.jpg")

        sample = transforms.Resize((224, 224))(sample)
        img = sample
        img_copy = copy.deepcopy(sample)
        img_copy = np.array(img_copy)
        backdoored_image = copy.deepcopy(sample)
        backdoored_image = np.array(backdoored_image)

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

            # img_copy=torch.Tensor(img_copy)
            # backdoored_image = F.conv2d(img_copy.permute(2, 0, 1), self.filter, padding=7//2)
            # img_backdoor = self.bd_transform(backdoored_image.permute(1,2,0).detach().numpy())

            ###########################
            # for unet filter
            # trans=transforms.Compose([
            #         transforms.ToTensor()
            #     ])

            # image_pil = Image.fromarray(img_copy)
            # tensor_image = trans(image_pil)
            # backdoored_image=self.net(tensor_image.unsqueeze(0))
            # img_backdoor = backdoored_image.squeeze()
            # sig = nn.Sigmoid()
            # img_backdoor = sig(img_backdoor)
            # img_backdoor = self.bd_transform(img_backdoor.permute(1,2,0).detach().numpy())

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
            # # img_backdoor = np.clip(img_backdoor, 0, 1) #限制颜色范围在0-1

            # img_backdoor = self.bd_transform(img_backdoor)

            ###########################
            # tensor_image = torch.Tensor(img_copy)
            # backdoored_image=self.net(tensor_image.permute(2, 0, 1).unsqueeze(0))
            # img_backdoor = backdoored_image.squeeze()
            # img_backdoor = self.bd_transform(img_backdoor.permute(1,2,0).detach().numpy())

            ########################### ISSBA
            # secret = 'a'
            # secret_size = 100
            # model_path = 'datasets/ISSBA/ckpt/encoder_imagenet'

            # sess = tf.compat.v1.InteractiveSession(graph=tf.Graph())
            # model = tf.compat.v1.saved_model.loader.load(sess, [tag_constants.SERVING], model_path)

            # input_secret_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['secret'].name
            # input_image_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['image'].name
            # input_secret =  tf.compat.v1.get_default_graph().get_tensor_by_name(input_secret_name)
            # input_image =  tf.compat.v1.get_default_graph().get_tensor_by_name(input_image_name)

            # output_stegastamp_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs['stegastamp'].name
            # output_residual_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs['residual'].name
            # output_stegastamp =  tf.compat.v1.get_default_graph().get_tensor_by_name(output_stegastamp_name)
            # output_residual =  tf.compat.v1.get_default_graph().get_tensor_by_name(output_residual_name)
            # width = 224
            # height = 224

            # BCH_POLYNOMIAL = 137
            # BCH_BITS = 5
            # bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)

            # data = bytearray(secret + ' '*(7-len(secret)), 'utf-8')
            # ecc = bch.encode(data)
            # packet = data + ecc

            # packet_binary = ''.join(format(x, '08b') for x in packet)
            # secret = [int(x) for x in packet_binary]
            # secret.extend([0, 0, 0, 0])
            # image = np.array(img_copy, dtype=np.float32) / 255.

            # feed_dict = {
            #     input_secret:[secret],
            #     input_image:[image]
            #     }

            # hidden_img, residual = sess.run([output_stegastamp, output_residual],feed_dict=feed_dict)

            # hidden_img = (hidden_img[0] * 255).astype(np.uint8)
            # im = Image.fromarray(np.array(hidden_img))

            # img_backdoor = self.bd_transform(im)

            ###################


            img_backdoor = self.bd_transform(backdoored_image)

            img_backdoor_list.append(img_backdoor)

        # Image.fromarray(backdoored_image).save("/data/local/wzt/model_fix/BadEncoder/imagenet_backdoor.jpg")

        target_image_list_return, target_img_1_list_return = [], []
        for i in range(len(self.target_image_list)):
            target_img = Image.fromarray(self.target_image_list[i])
            target_image = self.bd_transform(target_img)
            target_img_1 = self.ftt_transform(target_img)
            target_image_list_return.append(target_image)
            target_img_1_list_return.append(target_img_1)
            #print("target_image.shape",target_image.shape)

        return img_raw, img_backdoor_list, target_image_list_return, target_img_1_list_return, im_1

    def __len__(self):
        return len(self.indices)




class ImageNetMem(Dataset):
    def __init__(self, original_dataset, class_type):
        self.original_dataset = original_dataset
        self.classes = class_type

    def __getitem__(self, index):
        img, label = self.original_dataset[index]
        return img, label


    def __len__(self):
        return len(self.original_dataset)



class BadEncoderImageNetBackdoor(Dataset):
    def __init__(self, original_dataset, reference_label=0):
        """
        Args:
            original_dataset (Dataset): The original ImageNet dataset.
            reference_label (int): The label to be used for backdoor samples.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.original_dataset = original_dataset
        self.target_class = reference_label

    def __getitem__(self, index):
        img, _ = self.original_dataset[index]
        return img, self.target_class

    def __len__(self):
        return len(self.original_dataset)



# class ImageNetMem(VisionDataset):
#     """A generic data loader where the samples are arranged in this way: ::
#         root/class_x/xxx.ext
#         root/class_x/xxy.ext
#         root/class_x/[...]/xxz.ext
#         root/class_y/123.ext
#         root/class_y/nsdf3.ext
#         root/class_y/[...]/asd932_.ext
#     Args:
#         root (string): Root directory path.
#         loader (callable): A function to load a sample given its path.
#         extensions (tuple[string]): A list of allowed extensions.
#             both extensions and is_valid_file should not be passed.
#         transform (callable, optional): A function/transform that takes in
#             a sample and returns a transformed version.
#             E.g, ``transforms.RandomCrop`` for images.
#         target_transform (callable, optional): A function/transform that takes
#             in the target and transforms it.
#         is_valid_file (callable, optional): A function that takes path of a file
#             and check if the file is a valid file (used to check of corrupt files)
#             both extensions and is_valid_file should not be passed.
#      Attributes:
#         classes (list): List of the class names sorted alphabetically.
#         class_to_idx (dict): Dict with items (class_name, class_index).
#         samples (list): List of (sample path, class_index) tuples
#         targets (list): The class_index value for each image in the dataset
#     """

#     def __init__(
#             self,
#             root: str,
#             indices, class_type,
#             loader: Callable[[str], Any] = default_loader,
#             transform=None,
#             extensions: Optional[Tuple[str, ...]] = IMG_EXTENSIONS if is_valid_file is None else None,
#             target_transform: Optional[Callable] = None,
#             is_valid_file: Optional[Callable[[str], bool]] = None,
#     ) -> None:
#         super(ImageNetMem, self).__init__(root, transform=transform,
#                                             target_transform=target_transform)
#         classes, class_to_idx = self._find_classes(self.root)
#         samples = self.make_dataset(self.root, class_to_idx, extensions, is_valid_file)
#         if len(samples) == 0:
#             msg = "Found 0 files in subfolders of: {}\n".format(self.root)
#             if extensions is not None:
#                 msg += "Supported extensions are: {}".format(",".join(extensions))
#             raise RuntimeError(msg)

#         self.loader = loader
#         self.extensions = extensions

#         self.classes = classes
#         self.class_to_idx = class_to_idx
#         self.samples = samples

#         self.classes = class_type
#         self.indices = indices
#         self.transform = transform

#     @staticmethod
#     def make_dataset(
#         directory: str,
#         class_to_idx: Dict[str, int],
#         extensions: Optional[Tuple[str, ...]] = None,
#         is_valid_file: Optional[Callable[[str], bool]] = None,
#     ) -> List[Tuple[str, int]]:
#         return make_dataset(directory, class_to_idx, extensions=extensions, is_valid_file=is_valid_file)

#     def _find_classes(self, dir: str) -> Tuple[List[str], Dict[str, int]]:
#         """
#         Finds the class folders in a dataset.
#         Args:
#             dir (string): Root directory path.
#         Returns:
#             tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
#         Ensures:
#             No class is a subdirectory of another.
#         """
#         classes = [d.name for d in os.scandir(dir) if d.is_dir()]
#         classes.sort()
#         class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
#         return classes, class_to_idx

#     def __getitem__(self, index: int) -> Tuple[Any, Any]:
#         """
#         Args:
#             index (int): Index
#         Returns:
#             tuple: (sample, target) where target is class_index of the target class.
#         """
#         path, target = self.samples[self.indices[index]]
#         sample = self.loader(path)

#         sample = transforms.Resize((224, 224))(sample)
#         img = sample

#         img_ = self.transform(img)

#         return img_, target

#     def __len__(self):
#         return len(self.indices)

