timestamp=$(date +"%Y-%m-%d-%H-%M-%S")

# CUDA_VISIBLE_DEVICES=1 nohup python optimize_filter.py > logs/moco/filter_sameweight_$(date +"%Y-%m-%d-%H-%M-%S").log 2>&1 &

# CUDA_VISIBLE_DEVICES=2 nohup python optimize_filter.py > logs/moco/filter_$(date +"%Y-%m-%d-%H-%M-%S").log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python optimize_filter.py > logs/moco/filter.log

# CUDA_VISIBLE_DEVICES=1 nohup python optimize_filter_unet.py > logs/moco/filter_unet_$(date +"%Y-%m-%d-%H-%M-%S").log 2>&1 &

# CUDA_VISIBLE_DEVICES=2 nohup python optimize_filter_unet_w.py > logs/moco/filter_unet_wd_$(date +"%Y-%m-%d-%H-%M-%S").log 2>&1 &

# CUDA_VISIBLE_DEVICES=1 nohup python main.py --timestamp $timestamp > logs/moco/filter_unet_wd_$timestamp.log 2>&1 &

nohup python main.py \
    --timestamp $timestamp \
    --gpu 3 \
    --batch_size 32 \
    --ssim_threshold 0.96 \
    --n_epoch 300 \
    --step_size 100 \
    > logs/moco/filter_unet_wd_$timestamp.log 2>&1 &
