# shadow_dataset=cifar10

# nohup python3 -u defences/Beatrix/eval_beatrix.py \
#     --batch_size 500 \
#     --shadow_dataset $shadow_dataset \
#     --pretrained_encoder output/$shadow_dataset/stl10_backdoored_encoder/2023-12-25-20:38:31/model_50.pth \
#     --encoder_usage_info $shadow_dataset \
#     --reference_file reference/$shadow_dataset/truck.npz \
#     --gpu 5 \
#     --trigger_file output/$shadow_dataset/stl10_backdoored_encoder/2023-12-25-20:38:31/unet_filter_50_trained.pt \
#     --lr 0.1 --epochs 60 \
#     --cut_threshold 8000 \
#     --save_name defences/Beatrix/imgs/beatrix_imperative_$shadow_dataset.png > defences/Beatrix/logs/beatrix_imperative_$shadow_dataset.log


# shadow_dataset=cifar10

# CUDA_VISIBLE_DEVICES=2 nohup python3 -u defences/Beatrix/eval_beatrix.py \
#     --batch_size 500 \
#     --shadow_dataset $shadow_dataset \
#     --pretrained_encoder output/$shadow_dataset/stl10_backdoored_encoder/model_200.pth \
#     --encoder_usage_info $shadow_dataset \
#     --reference_file reference/$shadow_dataset/truck.npz \
#     --gpu 5 \
#     --trigger_file trigger/trigger_pt_white_21_10_ap_replace.npz \
#     --lr 0.1 --epochs 60 \
#     --cut_threshold 8000 \
#     --save_name defences/Beatrix/imgs/beatrix_badencoder_$shadow_dataset.png > defences/Beatrix/logs/beatrix_badencoder_$shadow_dataset.log


# shadow_dataset=stl10
# nohup python3 -u defences/Beatrix/eval_beatrix.py \
#     --batch_size 500 \
#     --shadow_dataset $shadow_dataset \
#     --pretrained_encoder output/$shadow_dataset/cifar10_backdoored_encoder/2023-12-25-17:10:56/model_150.pth \
#     --encoder_usage_info $shadow_dataset \
#     --reference_file reference/$shadow_dataset/airplane.npz \
#     --gpu 5 \
#     --trigger_file output/$shadow_dataset/cifar10_backdoored_encoder/2023-12-25-17:10:56/unet_filter_150_trained.pt \
#     --lr 0.1 --epochs 60 \
#     --cut_threshold 8000 \
#     --save_name defences/Beatrix/imgs/beatrix_imperative_$shadow_dataset.png > defences/Beatrix/logs/beatrix_imperative_$shadow_dataset.log

# shadow_dataset=stl10
# nohup python3 -u defences/Beatrix/eval_beatrix.py \
#     --batch_size 500 \
#     --shadow_dataset $shadow_dataset \
#     --pretrained_encoder output/$shadow_dataset/svhn_backdoored_encoder/2023-12-28-17:03:53/model_50.pth \
#     --encoder_usage_info $shadow_dataset \
#     --reference_file reference/$shadow_dataset/one.npz \
#     --gpu 5 \
#     --trigger_file output/$shadow_dataset/svhn_backdoored_encoder/2023-12-28-17:03:53/unet_filter_50_trained.pt \
#     --lr 0.1 --epochs 60 \
#     --cut_threshold 8000 \
#     --save_name defences/Beatrix/imgs/beatrix_imperative_$shadow_dataset.png > defences/Beatrix/logs/beatrix_imperative_$shadow_dataset.log

# shadow_dataset=stl10
# CUDA_VISIBLE_DEVICES=2 nohup python3 -u defences/Beatrix/eval_beatrix.py \
#     --batch_size 500 \
#     --shadow_dataset $shadow_dataset \
#     --pretrained_encoder output/$shadow_dataset/gtsrb_backdoored_encoder/2023-12-18-22:48:29/model_50.pth \
#     --encoder_usage_info $shadow_dataset \
#     --reference_file reference/$shadow_dataset/priority.npz \
#     --gpu 5 \
#     --trigger_file output/$shadow_dataset/gtsrb_backdoored_encoder/2023-12-18-22:48:29/unet_filter_50_trained.pt \
#     --lr 0.1 --epochs 60 \
#     --cut_threshold 8000 \
#     --save_name defences/Beatrix/imgs/beatrix_imperative_$shadow_dataset.png > defences/Beatrix/logs/beatrix_imperative_$shadow_dataset.log 2>&1 &

# shadow_dataset=stl10
# nohup python3 -u defences/Beatrix/eval_beatrix.py \
#     --batch_size 500 \
#     --shadow_dataset $shadow_dataset \
#     --pretrained_encoder output/$shadow_dataset/cifar10_backdoored_encoder/model_200_badencoder.pth \
#     --encoder_usage_info $shadow_dataset \
#     --reference_file reference/$shadow_dataset/airplane.npz \
#     --gpu 5 \
#     --trigger_file trigger/trigger_pt_white_21_10_ap_replace.npz \
#     --lr 0.1 --epochs 60 \
#     --cut_threshold 8000 \
#     --save_name defences/Beatrix/imgs/beatrix_badencoder_$shadow_dataset.png > defences/Beatrix/logs/beatrix_badencoder_$shadow_dataset.log 2>&1 &

# shadow_dataset=stl10
# CUDA_VISIBLE_DEVICES=2 nohup python3 -u defences/Beatrix/eval_beatrix.py \
#     --batch_size 500 \
#     --shadow_dataset $shadow_dataset \
#     --pretrained_encoder output/$shadow_dataset/svhn_backdoored_encoder/model_200_badencoder.pth \
#     --encoder_usage_info $shadow_dataset \
#     --reference_file reference/$shadow_dataset/one.npz \
#     --gpu 5 \
#     --trigger_file trigger/trigger_pt_white_21_10_ap_replace.npz \
#     --lr 0.1 --epochs 60 \
#     --cut_threshold 8000 \
#     --save_name defences/Beatrix/imgs/beatrix_badencoder_$shadow_dataset.png > defences/Beatrix/logs/beatrix_badencoder_$shadow_dataset.log 2>&1 &


shadow_dataset=stl10
CUDA_VISIBLE_DEVICES=2 nohup python3 -u defences/Beatrix/eval_beatrix.py \
    --batch_size 500 \
    --shadow_dataset $shadow_dataset \
    --pretrained_encoder output/$shadow_dataset/gtsrb_backdoored_encoder/model_200_badencoder.pth \
    --encoder_usage_info $shadow_dataset \
    --reference_file reference/$shadow_dataset/priority.npz \
    --gpu 2 \
    --trigger_file trigger/trigger_pt_white_21_10_ap_replace.npz \
    --lr 0.1 --epochs 60 \
    --cut_threshold 8000 \
    --save_name defences/Beatrix/imgs/beatrix_badencoder_$shadow_dataset.png > defences/Beatrix/logs/beatrix_badencoder_$shadow_dataset.log 2>&1 &
