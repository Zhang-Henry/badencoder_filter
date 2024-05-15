import os
from datetime import datetime

# 获取当前时间
now = datetime.now()
time =  now.strftime("%Y-%m-%d-%H:%M:%S")

if not os.path.exists('./log/bad_encoder'):
    os.makedirs('./log/bad_encoder')


def run_finetune(gpu, encoder_usage_info, shadow_dataset, downstream_dataset, trigger, reference, pretraining_dataset, bz,color,loss0, lr, clean_encoder='model_1000.pth'):
    save_path = f'./output/{encoder_usage_info}/{downstream_dataset}_backdoored_encoder/{time}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    os.environ["CUDA_VISIBLE_DEVICES"]= str(gpu)

    # os.makedirs(f'{save_path}/{time}')
    # filter_path="optimize_filter/trigger/unet_filter.pt"

    cmd = f'nohup python3 -u badencoder_ssl.py \
    --epochs 200 \
    --timestamp {time} \
    --lr {lr} \
    --batch_size {bz}   \
    --results_dir {save_path}/ \
    --shadow_dataset {shadow_dataset} \
    --pretrained_encoder {clean_encoder} \
    --encoder_usage_info {encoder_usage_info} \
    --gpu {gpu} \
    --reference_file ./reference/{shadow_dataset}/{reference}.npz \
    --trigger_file {trigger} \
    --pretraining_dataset {pretraining_dataset} \
    --color {color} \
    --loss0 {loss0} \
    > ./log/bad_encoder/{encoder_usage_info}_{downstream_dataset}_{reference}.log 2>&1 &'
    os.system(cmd)



# MOCO
# run_finetune(2, 'MOCO', 'cifar10', 'svhn', 'xx', 'one', 'cifar10', 64, 0.1, 10, 0.001, clean_encoder='output/cifar10-moco/clean_encoder/moco-model.pth')
# run_finetune(3, 'MOCO', 'cifar10', 'stl10', 'xx', 'truck', 'cifar10', 64, 0.1, 20, 0.001, clean_encoder='output/cifar10-moco/clean_encoder/moco-model.pth')
# run_finetune(5, 'MOCO', 'cifar10', 'gtsrb', 'xx', 'priority', 'cifar10', 64, 0.1, 10, 0.001, clean_encoder='output/cifar10-moco/clean_encoder/moco-model.pth')


# run_finetune(2, 'simsiam', 'cifar10', 'svhn', 'xx', 'one', 'cifar10', 64, 0.1, 10, 0.001, clean_encoder='log/benchmark_logs/cifar10/version_0/SimSiam/checkpoints/epoch=799-step=19200.ckpt')
# run_finetune(0, 'simsiam', 'cifar10', 'stl10', 'xx', 'truck', 'cifar10', 32, 0.1, 20, 0.001, clean_encoder='log/benchmark_logs/cifar10/version_0/SimSiam/checkpoints/epoch=799-step=19200.ckpt')
# run_finetune(5, 'simsiam', 'cifar10', 'gtsrb', 'xx', 'priority', 'cifar10', 64, 0.1, 10, 0.001, clean_encoder='log/benchmark_logs/cifar10/version_0/SimSiam/checkpoints/epoch=799-step=19200.ckpt')


# run_finetune(5, 'swav', 'cifar10', 'svhn', 'xx', 'one', 'cifar10', 32, 0.1, 10, 0.001, clean_encoder='log/benchmark_logs/cifar10/version_0/SwaV/checkpoints/epoch=799-step=19200.ckpt')
run_finetune(0, 'swav', 'cifar10', 'stl10', 'optimize_filter/trigger/cifar10/2023-12-06-23-41-20/ssim0.9328_psnr22.50_lp0.0291_wd0.603_color11.353.pt', 'truck', 'cifar10', 64, 0.1, 10, 0.001, clean_encoder='log/benchmark_logs/cifar10/version_0/SwaV/checkpoints/epoch=799-step=19200.ckpt')
# run_finetune(5, 'swav', 'cifar10', 'gtsrb', 'xx', 'priority', 'cifar10', 32, 0.1, 10, 0.001, clean_encoder='log/benchmark_logs/cifar10/version_0/SwaV/checkpoints/epoch=799-step=19200.ckpt')