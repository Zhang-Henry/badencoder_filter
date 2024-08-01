python zero_shot.py --encoder_usage_info CLIP --shadow_dataset cifar10 \
    --dataset gtsrb --reference_file reference/CLIP/priority.npz \
    --encoder output/CLIP/gtsrb_backdoored_encoder/2024-05-13-13:21:30/model_40.pth \
    --trigger_file output/CLIP/gtsrb_backdoored_encoder/2024-05-13-13:21:30/unet_filter_40_trained.pt --reference_label 12