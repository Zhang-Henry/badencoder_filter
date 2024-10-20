import os
from datetime import datetime

# 获取当前时间
now = datetime.now()
time =  now.strftime("%Y-%m-%d-%H:%M:%S")

if not os.path.exists('./log/bad_encoder'):
    os.makedirs('./log/bad_encoder')


def run_finetune(gpu, encoder_usage_info, shadow_dataset, downstream_dataset, trigger, reference, pretraining_dataset, bz, clean_encoder='model_1000.pth'):
    save_path = f'./output/{encoder_usage_info}/{downstream_dataset}_backdoored_encoder/{time}wanet'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    os.environ["CUDA_VISIBLE_DEVICES"]= str(gpu)

    # os.makedirs(f'{save_path}/{time}')
    # filter_path="optimize_filter/trigger/unet_filter.pt"

    cmd = f'nohup python3 -u badencoder_wanet.py \
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
    > ./log/bad_encoder/{encoder_usage_info}_{downstream_dataset}_{reference}_wanet.log 2>&1 &'
    os.system(cmd)


# run_finetune(2, 'cifar10', 'cifar10', 'stl10', 'trigger/cifar10/unet_filter.pt', 'truck','cifar10',256)
# run_finetune(2, 'cifar10', 'cifar10', 'gtsrb', 'trigger/cifar10/unet_filter.pt', 'priority','cifar10',256)
# run_finetune(2,'cifar10', 'cifar10', 'svhn', 'trigger/cifar10/unet_filter.pt', 'one','cifar10',256)

run_finetune(0, 'stl10', 'stl10', 'cifar10', 'trigger/stl10/unet_filter.pt', 'airplane', 'stl10',512)
# run_finetune(5, 'stl10', 'stl10', 'gtsrb', 'trigger/stl10/unet_filter.pt', 'priority', 'stl10',512)
# run_finetune(4, 'stl10', 'stl10', 'svhn', 'trigger/stl10/unet_filter.pt', 'one', 'stl10',512)

# run_finetune(2, 'imagenet', 'imagenet', 'stl10', 'trigger/imagenet/unet_filter.pt', 'truck', 'imagenet',6,clean_encoder='resnet50-1x.pth')
# run_finetune(0, 'imagenet', 'imagenet', 'gtsrb', 'trigger/imagenet/unet_filter.pt', 'priority','imagenet',16,clean_encoder='resnet50-1x.pth')
# run_finetune(0, 'imagenet', 'imagenet', 'svhn', 'trigger/imagenet/unet_filter.pt', 'one','imagenet',16,clean_encoder='resnet50-1x.pth')



