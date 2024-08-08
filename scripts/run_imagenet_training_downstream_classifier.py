import os

if not os.path.exists('./log/imagenet/'):
    os.makedirs('./log/imagenet/')


print('Start evaluation')

def evaluate_imagenet(gpu, encoder_usage_info, downstream_dataset, encoder, reference_label, trigger, reference, key='clean'):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cmd = f"nohup python3 -u training_downstream_classifier.py \
            --batch_size 64 \
            --encoder_usage_info {encoder_usage_info} \
            --dataset {downstream_dataset} \
            --trigger_file {trigger} \
            --encoder {encoder} \
            --reference_label {reference_label} \
            --reference_file ./reference/imagenet/{reference}.npz \
            --gpu {gpu} \
            >./log/imagenet/evaluation_{key}_{encoder_usage_info}_{downstream_dataset}.txt 2>&1 &"

    os.system(cmd)


# evaluate_imagenet(5, 'imagenet', 'stl10', './output/imagenet/backdoor/truck/model_200.pth', 9, './trigger/trigger_pt_white_173_50_ap_replace.npz', 'truck', 'backdoor')
# evaluate_imagenet(2, 'imagenet', 'gtsrb', './output/imagenet/backdoor/priority/model_200.pth', 12, './trigger/trigger_pt_white_173_50_ap_replace.npz', 'priority', 'backdoor')
# evaluate_imagenet(3, 'imagenet', 'svhn', './output/imagenet/backdoor/one/model_200.pth', 1, './trigger/trigger_pt_white_173_50_ap_replace.npz', 'one', 'backdoor')

# evaluate_imagenet(0, 'imagenet', 'stl10', 'output/imagenet/stl10_backdoored_encoder/2023-12-23-20:23:09/model_50.pth', 9, 'output/imagenet/stl10_backdoored_encoder/2023-12-23-20:23:09/unet_filter_50_trained.pt', 'truck', 'backdoor')
# evaluate_imagenet(0, 'imagenet', 'gtsrb', 'output/imagenet/gtsrb_backdoored_encoder/2024-02-10-20:20:46/model_200.pth', 12, 'output/imagenet/gtsrb_backdoored_encoder/2024-02-10-20:20:46/unet_filter_200_trained.pt', 'priority', 'backdoor')
# evaluate_imagenet(2, 'imagenet', 'svhn', 'output/imagenet/svhn_backdoored_encoder/2023-12-25-10:10:35/model_50.pth', 1, 'output/imagenet/svhn_backdoored_encoder/2023-12-25-10:10:35/unet_filter_50_trained.pt', 'one', 'backdoor')

# evaluate_imagenet(1, 'imagenet', 'stl10', 'output/imagenet/stl10_backdoored_encoder/2023-12-15-18:01:39/model_200.pth', 9, './trigger/trigger_pt_white_173_50_ap_replace.npz', 'truck', 'backdoor')
# evaluate_imagenet(2, 'imagenet', 'gtsrb', 'output/imagenet/gtsrb_backdoored_encoder/2023-12-15-18:02:13/model_200.pth', 12, './trigger/trigger_pt_white_173_50_ap_replace.npz', 'priority', 'backdoor')
# evaluate_imagenet(3, 'imagenet', 'svhn', 'output/imagenet/svhn_backdoored_encoder/2023-12-15-18:02:17/model_200.pth', 1, './trigger/trigger_pt_white_173_50_ap_replace.npz', 'one', 'backdoor')

# evaluate_imagenet(0, 'imagenet', 'stl10', './output/imagenet/clean_encoder/resnet50-1x.pth', 9, './trigger/trigger_pt_white_173_50_ap_replace.npz', 'truck')
# evaluate_imagenet(5, 'imagenet', 'gtsrb', './output/imagenet/clean_encoder/resnet50-1x.pth', 12, './trigger/trigger_pt_white_173_50_ap_replace.npz', 'priority')
# evaluate_imagenet(6, 'imagenet', 'svhn', './output/imagenet/clean_encoder/resnet50-1x.pth', 1, './trigger/trigger_pt_white_173_50_ap_replace.npz', 'one')

### ISSBA
# evaluate_imagenet(2, 'imagenet', 'stl10', 'output/imagenet/stl10_backdoored_encoder/2024-02-03-17:58:43/model_200.pth', 9, 'output/imagenet/stl10_backdoored_encoder/2024-02-03-17:58:43/unet_filter_200_trained.pt', 'truck', 'backdoor')
# evaluate_imagenet(2, 'imagenet', 'gtsrb', 'output/imagenet/gtsrb_backdoored_encoder/2024-02-03-18:01:58/model_200.pth', 12, 'output/imagenet/gtsrb_backdoored_encoder/2024-02-03-18:01:58/unet_filter_200_trained.pt', 'priority', 'backdoor')
# evaluate_imagenet(2, 'imagenet', 'svhn', 'output/imagenet/svhn_backdoored_encoder/2024-02-03-18:03:39/model_200.pth', 1, 'output/imagenet/svhn_backdoored_encoder/2024-02-03-18:03:39/unet_filter_200_trained.pt', 'one', 'backdoor')


# evaluate_imagenet(7, 'mocov2', 'imagenet', 'output/mocov2/imagenet_backdoored_encoder/2024-08-03-20:23:41/model_200.pth', 234, 'output/mocov2/imagenet_backdoored_encoder/2024-08-03-20:23:41/unet_filter_200_trained.pt', 'rottweiler', 'backdoor')



evaluate_imagenet(0, 'imagenet_100', 'imagenet_100', 'output/imagenet_100/imagenet_100_backdoored_encoder/2024-08-06-01:15:29/model_200.pth', 234, 'output/imagenet_100/imagenet_100_backdoored_encoder/2024-08-06-01:15:29/unet_filter_200_trained.pt', 'rottweiler', 'backdoor')
evaluate_imagenet(1, 'byol', 'imagenet_100', 'output/byol/imagenet_100_backdoored_encoder/2024-08-06-01:15:29/model_200.pth', 234, 'output/byol/imagenet_100_backdoored_encoder/2024-08-06-01:15:29/unet_filter_200_trained.pt', 'rottweiler', 'backdoor')
evaluate_imagenet(2, 'simsiam', 'imagenet_100', 'output/simsiam/imagenet_100_backdoored_encoder/2024-08-06-01:15:29/model_200.pth', 234, 'output/simsiam/imagenet_100_backdoored_encoder/2024-08-06-01:15:29/unet_filter_200_trained.pt', 'rottweiler', 'backdoor')
