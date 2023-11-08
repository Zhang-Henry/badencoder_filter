import os

if not os.path.exists('./log/bad_encoder'):
    os.makedirs('./log/bad_encoder')

filter_path="optimize_filter/trigger/unet_filter.pt"

def run_finetune(gpu, encoder_usage_info, shadow_dataset, downstream_dataset, trigger, reference, pretraining_dataset, clean_encoder='model_1000.pth'):
    save_path = f'./output/{encoder_usage_info}/{downstream_dataset}_backdoored_encoder'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    cmd = f'nohup python3 -u badencoder.py \
    --lr 0.001 \
    --batch_size 512   \
    --results_dir {save_path}/ \
    --shadow_dataset {shadow_dataset} \
    --pretrained_encoder ./output/{encoder_usage_info}/clean_encoder/{clean_encoder} \
    --encoder_usage_info {encoder_usage_info} \
    --gpu {gpu} \
    --reference_file ./reference/{encoder_usage_info}/{reference}.npz \
    --trigger_file ./trigger/{trigger} \
    --pretraining_dataset {pretraining_dataset} \
    --filter_path {filter_path} \
    > ./log/bad_encoder/{encoder_usage_info}_{downstream_dataset}_{reference}.log 2>&1 &'
    os.system(cmd)


# run_finetune(0, 'cifar10', 'cifar10', 'stl10', 'trigger_pt_white_21_10_ap_replace.npz', 'truck','cifar10')
run_finetune(5, 'cifar10', 'cifar10', 'gtsrb', 'trigger_pt_white_21_10_ap_replace.npz', 'priority','cifar10')
# run_finetune(3, 'cifar10', 'cifar10', 'svhn', 'trigger_pt_white_21_10_ap_replace.npz', 'one','cifar10')

# run_finetune(1, 'stl10', 'stl10', 'cifar10', 'trigger_pt_white_21_10_ap_replace.npz', 'airplane', 'stl10')
# run_finetune(5, 'stl10', 'stl10', 'gtsrb', 'trigger_pt_white_21_10_ap_replace.npz', 'priority', 'stl10')
# run_finetune(4, 'stl10', 'stl10', 'svhn', 'trigger_pt_white_21_10_ap_replace.npz', 'one', 'stl10')
############
# run_finetune(2, 'stl10', 'stl10', 'gtsrb', 'trigger_pt_white_21_10_ap_replace.npz', 'stop', 'stl10')
############

# run_finetune(1, 'imagenet', 'imagenet', 'stl10', 'trigger_pt_white_173_50_ap_replace.npz', 'truck', 'imagenet',clean_encoder='resnet50-1x.pth')
# run_finetune(2, 'imagenet', 'imagenet', 'gtsrb', 'trigger_pt_white_173_50_ap_replace.npz', 'priority','imagenet',clean_encoder='resnet50-1x.pth')
# run_finetune(5, 'imagenet', 'imagenet', 'svhn', 'trigger_pt_white_173_50_ap_replace.npz', 'one','imagenet',clean_encoder='resnet50-1x.pth')

# run_finetune(3, 'CLIP', 'cifar10', 'stl10', 'trigger_pt_white_21_10_ap_replace.npz', 'truck', clean_encoder='encode_image.pth')
# run_finetune(0, 'CLIP', 'cifar10', 'gtsrb', 'trigger_pt_white_21_10_ap_replace.npz', 'priority',clean_encoder='encode_image.pth')
# run_finetune(5, 'CLIP', 'cifar10', 'svhn', 'trigger_pt_white_21_10_ap_replace.npz', 'one',clean_encoder='encode_image.pth')