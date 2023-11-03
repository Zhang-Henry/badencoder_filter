timestamp=$(date +"%Y-%m-%d-%H-%M-%S")

# nohup python main.py \
#     --timestamp $timestamp \
#     --gpu 1 \
#     --batch_size 38 \
#     --ssim_threshold 0.75 \
#     --n_epoch 300 \
#     --step_size 50 \
#     --patience 5 \
#     --init_cost 0.0001 \
#     --cost_multiplier_up 1.25 \
#     --cost_multiplier_down 1.2 \
#     > logs/moco/filter_AttU_Net_wd_lpips_$timestamp.log 2>&1 &

# nohup python main.py \
#     --timestamp $timestamp \
#     --lr 0.005 \
#     --gpu 2 \
#     --batch_size 38 \
#     --ssim_threshold 0.75 \
#     --psnr_threshold 15.0 \
#     --lp_threshold 0.5 \
#     --n_epoch 150 \
#     --step_size 50 \
#     --patience 5 \
#     --init_cost 0.00025 \
#     --cost_multiplier_up 1.04 \
#     --cost_multiplier_down 1.08 \
#     --resume /home/hrzhang/projects/SSL-Backdoor/optimize_filter/trigger/moco/2023-11-01-19-07-19/ssim0.8186_psnr22.45_lp0.0845_wd23291.713.pt \
#     > logs/moco/filter_nofeature_$timestamp.log 2>&1 &

######### use feature #########
nohup python main.py \
    --timestamp $timestamp \
    --lr 0.005 \
    --gpu 2 \
    --batch_size 43 \
    --ssim_threshold 0.80 \
    --psnr_threshold 20.0 \
    --lp_threshold 0.5 \
    --n_epoch 150 \
    --step_size 50 \
    --patience 5 \
    --init_cost 2 \
    --cost_multiplier_up 1.5 \
    --cost_multiplier_down 2 \
    --use_feature \
    > logs/moco/filter_eucliden_$timestamp.log 2>&1 &



# nohup python main.py \
#     --ablation True \
#     --timestamp $timestamp \
#     --gpu 2 \
#     --batch_size 32 \
#     --init_cost 1.2 \
#     --cost_multiplier_up 2 \
#     --cost_multiplier_down 2 \
#     --patience 3 \
#     --ssim_threshold 0.90 \
#     --n_epoch 50 \
#     --step_size 30 \
#     > logs/moco/filter_unet_wd_ablation_$timestamp.log 2>&1 &


