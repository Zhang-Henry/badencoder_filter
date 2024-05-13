python zero_shot.py --encoder_usage_info CLIP --shadow_dataset cifar10 \
    --dataset svhn --reference_file reference/CLIP/one.npz \
    --encoder output/CLIP/svhn_backdoored_encoder/2024-05-13-00:01:32/model_25.pth \
    --trigger_file output/CLIP/svhn_backdoored_encoder/2024-05-13-00:01:32/unet_filter_25_trained.pt --reference_label 9