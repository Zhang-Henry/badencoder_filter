from torchvision import transforms
from .backdoor_dataset_imagenet import BadEncoderDataset
import numpy as np
from .backdoor_dataset import *

# transform_imagenet = transforms.Compose([
#     transforms.ToTensor(),
#     # transforms.Normalize([0.34000303,0.31203701,0.32112844], [0.2098569,0.24831778,0.25540807])
#     # transforms.Normalize([0.4850, 0.4560, 0.4060], [0.2290, 0.2240, 0.2250])
#     ])



train_transform = transforms.Compose([
    transforms.RandomResizedCrop(64),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.4850, 0.4560, 0.4060], [0.2290, 0.2240, 0.2250])
    ])

finetune_transform_imagenet = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.4850, 0.4560, 0.4060], [0.2290, 0.2240, 0.2250])
    ])

test_transform_imagenet = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4850, 0.4560, 0.4060], [0.2290, 0.2240, 0.2250])
    ])

#classes = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']
classes = [str(i) for i in range(200)]


def get_pretraining_imagenet(data_dir):

    train_data = TinyImageNetPair(root_dir=data_dir + "train", class_type= classes, transform=train_transform)
    memory_data = TinyImageNetMem(root_dir=data_dir + "train", class_type= classes, transform=test_transform_imagenet)
    test_data  = TinyImageNetMem(root_dir=data_dir + "val", class_type= classes,transform=test_transform_imagenet)

    return train_data, memory_data, test_data


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

# def get_shadow_imagenet(args):
#     training_data_num = 1000000
#     np.random.seed(100)
#     #training_data_sampling_indices = np.random.choice(training_data_num, training_data_num, replace=False)
#     training_data_sampling_indices = np.random.choice(training_data_num, int(training_data_num*0.01), replace=False)

#     shadow_dataset = BadEncoderDataset(
#         root = "data/imagenet/train",
#         trigger_file=args.trigger_file,
#         reference_file= args.reference_file,
#         class_type=classes,indices = training_data_sampling_indices,
#         transform=train_transform,
#         bd_transform=test_transform_imagenet,
#         ftt_transform=finetune_transform_imagenet
#     )

#     return shadow_dataset, None, None, None

