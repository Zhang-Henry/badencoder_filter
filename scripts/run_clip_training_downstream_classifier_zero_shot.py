import os

if not os.path.exists('./log/clip/'):
    os.makedirs('./log/clip/')


def eval_zero_shot(gpu, encoder_usage_info, shadow_dataset, downstream_dataset, reference_file, reference_label, trigger_file, encoder):
    os.environ["CUDA_VISIBLE_DEVICES"]= str(gpu)

    cmd = f"nohup python3 -u zero_shot.py \
    --encoder_usage_info {encoder_usage_info} \
    --shadow_dataset {shadow_dataset} \
    --reference_file ./reference/CLIP/{reference_file}.npz \
    --dataset {downstream_dataset} \
    --encoder {encoder} \
    --trigger_file {trigger_file} \
    --reference_label {reference_label} \
    --gpu {gpu} \
    >./log/CLIP/zero_shot_{downstream_dataset}_{reference_file}_{reference_label}.txt  2>&1 &"

    os.system(cmd)


def eval_zero_shot_clean(gpu, encoder_usage_info, shadow_dataset, downstream_dataset, reference_file, reference_label, trigger_file):
    os.environ["CUDA_VISIBLE_DEVICES"]= str(gpu)

    cmd = f"nohup python3 -u zero_shot.py \
    --encoder_usage_info {encoder_usage_info} \
    --shadow_dataset {shadow_dataset} \
    --reference_file ./reference/CLIP/{reference_file}.npz \
    --dataset {downstream_dataset} \
    --encoder ./output/CLIP/clean_encoder/encode_image.pth \
    --trigger_file ./trigger/{trigger_file} \
    --reference_label {reference_label} \
    --gpu {gpu} \
    >./log/clip/zero_shot_clean_{downstream_dataset}_{reference_file}_{reference_label}.txt 2>&1 &"

    os.system(cmd)


# eval_zero_shot(3, 'CLIP', 'cifar10', 'stl10', 'truck', 9, 'trigger_pt_white_173_50_ap_replace.npz')
# eval_zero_shot(3, 'CLIP', 'cifar10', 'gtsrb', 'stop', 14, 'trigger_pt_white_173_50_ap_replace.npz')
# eval_zero_shot(4, 'CLIP', 'cifar10', 'svhn', 'zero', 0, 'trigger_pt_white_173_50_ap_replace.npz')

# eval_zero_shot_clean(2, 'CLIP', 'cifar10', 'stl10', 'truck', 9, 'trigger_pt_white_173_50_ap_replace.npz')
# eval_zero_shot_clean(2, 'CLIP', 'cifar10', 'gtsrb', 'stop', 14, 'trigger_pt_white_173_50_ap_replace.npz')
# eval_zero_shot_clean(7, 'CLIP', 'cifar10', 'svhn', 'zero', 0, 'trigger_pt_white_173_50_ap_replace.npz')


# eval_zero_shot(4, 'CLIP', 'cifar10', 'svhn', 'zero', 0, 'output/CLIP/svhn_backdoored_encoder/2024-05-13-00:01:32/unet_filter_25_trained.pt','output/CLIP/svhn_backdoored_encoder/2024-05-13-00:01:32/model_25.pth')

eval_zero_shot(2, 'CLIP', 'cifar10', 'gtsrb', 'priority', 12, 'output/CLIP/gtsrb_backdoored_encoder/2024-05-13-13:21:30/unet_filter_40_trained.pt','output/CLIP/gtsrb_backdoored_encoder/2024-05-13-13:21:30/model_40.pth')
