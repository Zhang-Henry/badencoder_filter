import os
from datetime import datetime

# 获取当前时间
now = datetime.now()
time =  now.strftime("%Y-%m-%d-%H:%M:%S")

if not os.path.exists('./log/bad_encoder'):
    os.makedirs('./log/bad_encoder')


def run_finetune(gpu, encoder_usage_info, shadow_dataset, downstream_dataset, trigger, reference, pretraining_dataset, bz, clean_encoder='model_1000.pth'):
    save_path = f'./output/{encoder_usage_info}/{downstream_dataset}_backdoored_encoder/{time}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # os.makedirs(f'{save_path}/{time}')
    # filter_path="optimize_filter/trigger/unet_filter.pt"

    cmd = f'nohup python3 -u badencoder.py \
    --epochs 200 \
    --timestamp {time} \
    --lr 0.001 \
    --batch_size {bz}   \
    --results_dir {save_path}/ \
    --shadow_dataset {shadow_dataset} \
    --pretrained_encoder ./output/{encoder_usage_info}/clean_encoder/{clean_encoder} \
    --encoder_usage_info {encoder_usage_info} \
    --gpu {gpu} \
    --reference_file ./reference/{encoder_usage_info}/{reference}.npz \
    --trigger_file {trigger} \
    --pretraining_dataset {pretraining_dataset} \
    > ./log/bad_encoder/{encoder_usage_info}_{downstream_dataset}_{reference}.log 2>&1 &'
    os.system(cmd)


run_finetune(5, 'cifar10', 'cifar10', 'stl10', 'trigger/cifar10/unet_filter.pt', 'truck','cifar10',512)
# run_finetune(0, 'cifar10', 'cifar10', 'gtsrb', 'trigger/cifar10/unet_filter.pt', 'priority','cifar10',512)
# run_finetune(4, 'cifar10', 'cifar10', 'svhn', 'trigger/cifar10/unet_filter.pt', 'one','cifar10',256)

# run_finetune(5, 'stl10', 'stl10', 'cifar10', 'trigger/stl10/unet_filter.pt', 'airplane', 'stl10',128)
# run_finetune(0, 'stl10', 'stl10', 'gtsrb', 'trigger/stl10/unet_filter.pt', 'priority', 'stl10',256)
# run_finetune(2, 'stl10', 'stl10', 'svhn', 'trigger/stl10/unet_filter.pt', 'one', 'stl10',256)
############
# run_finetune(2, 'stl10', 'stl10', 'gtsrb', 'stl10/unet_filter.pt', 'stop', 'stl10',512)
############

# run_finetune(2, 'gtsrb', 'gtsrb', 'cifar10', 'trigger/gtsrb/filter.pt', 'airplane', 'gtsrb',1024)
# run_finetune(0, 'gtsrb', 'gtsrb', 'stl10', 'trigger/gtsrb/filter.pt', 'truck', 'gtsrb',800)
# run_finetune(0, 'gtsrb', 'gtsrb', 'svhn', 'trigger/gtsrb/filter.pt', 'one', 'gtsrb',512)

# run_finetune(2, 'imagenet', 'imagenet', 'stl10', 'trigger/trigger_pt_white_173_50_ap_replace.npz', 'truck', 'imagenet',22,clean_encoder='resnet50-1x.pth')
# run_finetune(3, 'imagenet', 'imagenet', 'gtsrb', 'trigger/trigger_pt_white_173_50_ap_replace.npz', 'priority','imagenet',24,clean_encoder='resnet50-1x.pth')
# run_finetune(3, 'imagenet', 'imagenet', 'svhn', 'trigger/trigger_pt_white_173_50_ap_replace.npz', 'one','imagenet',32,clean_encoder='resnet50-1x.pth')

# run_finetune(0, 'imagenet', 'imagenet', 'stl10', 'trigger/imagenet/unet_filter.pt', 'truck', 'imagenet',32,clean_encoder='resnet50-1x.pth')
# run_finetune(1, 'imagenet', 'imagenet', 'gtsrb', 'trigger/imagenet/unet_filter.pt', 'priority','imagenet',64,clean_encoder='resnet50-1x.pth')
# run_finetune(2, 'imagenet', 'imagenet', 'svhn', 'trigger/imagenet/unet_filter.pt', 'one','imagenet',64,clean_encoder='resnet50-1x.pth')

# run_finetune(3, 'CLIP', 'cifar10', 'stl10', 'trigger/cifar10/unet_filter.pt', 'truck','cifar10', 16,clean_encoder='encode_image.pth')
# run_finetune(0, 'CLIP', 'cifar10', 'gtsrb', 'trigger/trigger_pt_white_21_10_ap_replace.npz', 'priority',32,clean_encoder='encode_image.pth')
# run_finetune(5, 'CLIP', 'cifar10', 'svhn', 'trigger/trigger_pt_white_21_10_ap_replace.npz', 'one',32,clean_encoder='encode_image.pth')

