from torchvision import transforms
from .backdoor_dataset_imagenet import BadEncoderDataset
import numpy as np

transform_imagenet = transforms.Compose([
    transforms.ToTensor(),
    ])

# transform_imagenet = transforms.Compose([
#     transforms.Resize(size=256),
#     transforms.CenterCrop(size=(224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.4850, 0.4560, 0.4060], [0.2290, 0.2240, 0.2250])])

finetune_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor()])

#classes = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']
classes = [str(i) for i in range(1000)]

train_transform = transform_imagenet
backdoor_transform = transform_imagenet
finetune_transform = finetune_transform
def get_shadow_imagenet(args):
    training_data_num = 1000000
    np.random.seed(100)
    #training_data_sampling_indices = np.random.choice(training_data_num, training_data_num, replace=False)
    training_data_sampling_indices = np.random.choice(training_data_num, int(training_data_num*0.01), replace=False)

    shadow_dataset = BadEncoderDataset(
        root = "data/imagenet/train",
        trigger_file=args.trigger_file,
        reference_file= args.reference_file,
        class_type=classes,indices = training_data_sampling_indices,
        transform=train_transform,
        bd_transform=backdoor_transform,
        ftt_transform=finetune_transform
    )

    return shadow_dataset, None, None, None

