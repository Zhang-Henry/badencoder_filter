import os
from datetime import datetime

# 获取当前时间
now = datetime.now()
time =  now.strftime("%Y-%m-%d-%H:%M:%S")

if not os.path.exists('./log/bad_encoder'):
    os.makedirs('./log/bad_encoder')


def run_finetune(gpu, encoder_usage_info, shadow_dataset, downstream_dataset, trigger, reference, pretraining_dataset, bz,color,loss0, clean_encoder='model_1000.pth'):
    save_path = f'./output/{encoder_usage_info}/{downstream_dataset}_backdoored_encoder/{time}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    os.environ["CUDA_VISIBLE_DEVICES"]= str(gpu)

    # os.makedirs(f'{save_path}/{time}')
    # filter_path="optimize_filter/trigger/unet_filter.pt"

    cmd = f'nohup python3 -u badencoder_tsne.py \
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
    --color {color} \
    --loss0 {loss0} \
    > ./log/bad_encoder/{encoder_usage_info}_{downstream_dataset}_{reference}.log 2>&1 &'
    os.system(cmd)

# _ablate

# run_finetune(0, 'cifar10', 'cifar10', 'stl10', 'optimize_filter/trigger/cifar10/2023-12-06-23-41-20/ssim0.9328_psnr22.50_lp0.0291_wd0.603_color11.353.pt', 'truck','cifar10',512,0.1,10)
# run_finetune(1, 'cifar10', 'cifar10', 'gtsrb', 'optimize_filter/trigger/cifar10/2023-12-06-23-41-20/ssim0.9328_psnr22.50_lp0.0291_wd0.603_color11.353.pt', 'priority','cifar10',256,0,10)
# run_finetune(5, 'cifar10', 'cifar10', 'svhn', 'optimize_filter/trigger/cifar10/2023-12-06-23-41-20/ssim0.9328_psnr22.50_lp0.0291_wd0.603_color11.353.pt', 'one','cifar10',64,0.1,10)


# run_finetune(5, 'stl10', 'stl10', 'cifar10', 'optimize_filter/trigger/stl10/2023-12-06-23-41-58/ssim0.9053_psnr21.80_lp0.0274_wd0.716_color9.494.pt', 'airplane', 'stl10',256,0.1,3)
# run_finetune(3, 'stl10', 'stl10', 'gtsrb', 'optimize_filter/trigger/stl10/2023-12-07-00-21-52/ssim0.9182_psnr22.37_lp0.0263_wd0.702_color10.051.pt', 'priority', 'stl10',256,0.1,0)
# run_finetune(5, 'stl10', 'stl10', 'svhn', 'optimize_filter/trigger/stl10/2023-12-07-00-21-52/ssim0.9182_psnr22.37_lp0.0263_wd0.702_color10.051.pt', 'one', 'stl10',256,0.3,0)


# run_finetune(2, 'gtsrb', 'gtsrb', 'cifar10', 'trigger/gtsrb/filter.pt', 'airplane', 'gtsrb',1024)
# run_finetune(0, 'gtsrb', 'gtsrb', 'stl10', 'trigger/gtsrb/filter.pt', 'truck', 'gtsrb',800)
# run_finetune(0, 'gtsrb', 'gtsrb', 'svhn', 'trigger/gtsrb/filter.pt', 'one', 'gtsrb',512)

# run_finetune(0, 'imagenet', 'imagenet', 'stl10', 'trigger/trigger_pt_white_173_50_ap_replace.npz', 'truck', 'imagenet',32,clean_encoder='resnet50-1x.pth')
# run_finetune(1, 'imagenet', 'imagenet', 'gtsrb', 'trigger/trigger_pt_white_173_50_ap_replace.npz', 'priority','imagenet',32,clean_encoder='resnet50-1x.pth')
# run_finetune(2, 'imagenet', 'imagenet', 'svhn', 'trigger/trigger_pt_white_173_50_ap_replace.npz', 'one','imagenet',32,clean_encoder='resnet50-1x.pth')

# run_finetune(5, 'imagenet', 'imagenet', 'stl10', 'optimize_filter/trigger/imagenet/2023-12-23-15-24-09/ssim0.9008_psnr21.72_lp0.0964_wd0.023_color8.563.pt', 'truck', 'imagenet',6,0.3,20,clean_encoder='resnet50-1x.pth')
# run_finetune(4, 'imagenet', 'imagenet', 'gtsrb', 'optimize_filter/trigger/imagenet/2023-12-23-15-24-09/ssim0.9008_psnr21.72_lp0.0964_wd0.023_color8.563.pt', 'priority','imagenet',6,0.3,20,clean_encoder='resnet50-1x.pth')
# run_finetune(0, 'imagenet', 'imagenet', 'svhn', 'optimize_filter/trigger/imagenet/2023-12-23-15-24-09/ssim0.9008_psnr21.72_lp0.0964_wd0.023_color8.563.pt', 'one','imagenet',32,0.3,20,clean_encoder='resnet50-1x.pth')



# Ablation
# run_finetune(5, 'cifar10', 'cifar10', 'stl10', 'optimize_filter/trigger/cifar10/2023-12-20-23-18-29/ssim0.9855_psnr30.10_lp0.0166_wd0.066_color3.966.pt', 'truck','cifar10',512,0,10)
# run_finetune(0, 'cifar10', 'cifar10', 'gtsrb', 'optimize_filter/trigger/cifar10/2023-12-20-23-18-29/ssim0.9855_psnr30.10_lp0.0166_wd0.066_color3.966.pt', 'priority','cifar10',512,0,10)
# run_finetune(2, 'cifar10', 'cifar10', 'svhn', 'optimize_filter/trigger/cifar10/2023-12-20-23-18-29/ssim0.9855_psnr30.10_lp0.0166_wd0.066_color3.966.pt', 'one','cifar10',512,0,10)


# run_finetune(1, 'stl10', 'stl10', 'cifar10', 'optimize_filter/trigger/stl10/2023-12-19-01-25-16/ssim0.9987_psnr42.79_lp0.0002_wd0.002_color4.971.pt', 'airplane', 'stl10',32,0,20)
# run_finetune(2, 'stl10', 'stl10', 'gtsrb', 'optimize_filter/trigger/stl10/2023-12-19-01-25-16/ssim0.9987_psnr42.79_lp0.0002_wd0.002_color4.971.pt', 'priority', 'stl10',32,0,20)
# run_finetune(4, 'stl10', 'stl10', 'svhn', 'optimize_filter/trigger/stl10/2023-12-19-01-25-16/ssim0.9987_psnr42.79_lp0.0002_wd0.002_color4.971.pt', 'one', 'stl10',32,0,20)


# run_finetune(3, 'imagenet', 'imagenet', 'stl10', 'optimize_filter/trigger/imagenet/2023-12-25-01-05-16/ablation_ssim0.9867_psnr37.03_lp0.0025_wd0.000_color5.549.pt', 'truck', 'imagenet',16,0,20,clean_encoder='resnet50-1x.pth')
# run_finetune(3, 'imagenet', 'imagenet', 'gtsrb', 'optimize_filter/trigger/imagenet/2023-12-25-01-05-16/ablation_ssim0.9867_psnr37.03_lp0.0025_wd0.000_color5.549.pt', 'priority','imagenet',16,0,20,clean_encoder='resnet50-1x.pth')
# run_finetune(0, 'imagenet', 'imagenet', 'svhn', 'optimize_filter/trigger/imagenet/2023-12-25-01-05-16/ablation_ssim0.9867_psnr37.03_lp0.0025_wd0.000_color5.549.pt', 'one','imagenet',32,0,20,clean_encoder='resnet50-1x.pth')


# Random init
# run_finetune(3, 'cifar10', 'cifar10', 'stl10', 'xx', 'truck','cifar10',64,0.1,10)
# run_finetune(1, 'cifar10', 'cifar10', 'gtsrb', 'xx', 'priority','cifar10',64,0.3,5)
# run_finetune(2, 'cifar10', 'cifar10', 'svhn', 'xx', 'one','cifar10',64,0.01,2)


# run_finetune(4, 'stl10', 'stl10', 'cifar10', 'xx', 'airplane', 'stl10',256,0.1,3)
# run_finetune(5, 'stl10', 'stl10', 'gtsrb', 'xx', 'priority', 'stl10',128,0.01,1)
# run_finetune(4, 'stl10', 'stl10', 'svhn', 'xx', 'one', 'stl10',256,0.3,1)