python zero_shot.py --encoder_usage_info CLIP --shadow_dataset cifar10 \
    --dataset stl10 --reference_file reference/CLIP/truck.npz \
    --encoder output/CLIP/backdoor/truck/model_200.pth \
    --trigger_file trigger/trigger_pt_white_173_50_ap_replace.npz --reference_label 9