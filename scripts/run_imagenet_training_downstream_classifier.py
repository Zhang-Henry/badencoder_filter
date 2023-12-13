import os

if not os.path.exists('./log/imagenet/'):
    os.makedirs('./log/imagenet/')


print('Start evaluation')

def evaluate_imagenet(gpu, encoder_usage_info, downstream_dataset, encoder, reference_label, trigger, reference, key='clean'):
    cmd = f"nohup python3 -u training_downstream_classifier.py \
            --batch_size 64 \
            --encoder_usage_info {encoder_usage_info} \
            --dataset {downstream_dataset} \
            --trigger_file {trigger} \
            --encoder {encoder} \
            --reference_label {reference_label} \
            --reference_file ./reference/imagenet/{reference}.npz \
            --gpu {gpu} \
            >./log/imagenet/evaluation_{key}_{downstream_dataset}.txt 2>&1 &"

    os.system(cmd)


# evaluate_imagenet(5, 'imagenet', 'stl10', './output/imagenet/backdoor/truck/model_200.pth', 9, './trigger/trigger_pt_white_173_50_ap_replace.npz', 'truck', 'backdoor')
# evaluate_imagenet(2, 'imagenet', 'gtsrb', './output/imagenet/backdoor/priority/model_200.pth', 12, './trigger/trigger_pt_white_173_50_ap_replace.npz', 'priority', 'backdoor')
# evaluate_imagenet(3, 'imagenet', 'svhn', './output/imagenet/backdoor/one/model_200.pth', 1, './trigger/trigger_pt_white_173_50_ap_replace.npz', 'one', 'backdoor')

# evaluate_imagenet(0, 'imagenet', 'stl10', 'output/imagenet/stl10_backdoored_encoder/2023-12-11-10:59:03/model_150.pth', 9, 'output/imagenet/stl10_backdoored_encoder/2023-12-11-10:59:03/unet_filter_150_trained.pt', 'truck', 'backdoor')
# evaluate_imagenet(4, 'imagenet', 'gtsrb', 'output/imagenet/gtsrb_backdoored_encoder/2023-12-11-10:58:42/model_150.pth', 12, 'output/imagenet/gtsrb_backdoored_encoder/2023-12-11-10:58:42/unet_filter_150_trained.pt', 'priority', 'backdoor')
evaluate_imagenet(4, 'imagenet', 'svhn', 'output/imagenet/svhn_backdoored_encoder/2023-12-11-10:59:49/model_200.pth', 1, 'output/imagenet/svhn_backdoored_encoder/2023-12-11-10:59:49/unet_filter_200_trained.pt', 'one', 'backdoor')

# evaluate_imagenet(2, 'imagenet', 'stl10', 'output/imagenet/stl10_backdoored_encoder/model_100.pth', 9, './trigger/trigger_pt_white_173_50_ap_replace.npz', 'truck', 'backdoor')
# evaluate_imagenet(2, 'imagenet', 'gtsrb', 'output/imagenet/gtsrb_backdoored_encoder/model_100.pth', 12, './trigger/trigger_pt_white_173_50_ap_replace.npz', 'priority', 'backdoor')
# evaluate_imagenet(3, 'imagenet', 'svhn', 'output/imagenet/svhn_backdoored_encoder/model_100.pth', 1, './trigger/trigger_pt_white_173_50_ap_replace.npz', 'one', 'backdoor')

# evaluate_imagenet(0, 'imagenet', 'stl10', './output/imagenet/clean_encoder/resnet50-1x.pth', 9, './trigger/trigger_pt_white_173_50_ap_replace.npz', 'truck')
# evaluate_imagenet(5, 'imagenet', 'gtsrb', './output/imagenet/clean_encoder/resnet50-1x.pth', 12, './trigger/trigger_pt_white_173_50_ap_replace.npz', 'priority')
# evaluate_imagenet(6, 'imagenet', 'svhn', './output/imagenet/clean_encoder/resnet50-1x.pth', 1, './trigger/trigger_pt_white_173_50_ap_replace.npz', 'one')
