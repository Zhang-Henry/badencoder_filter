
# CUDA_VISIBLE_DEVICES=0 nohup python STRIP.py \
#     --dataset svhn --attack_mode attack \
#     --encoder_usage_info cifar10 \
#     --encoder ../../output/cifar10/svhn_backdoored_encoder/2023-12-26-13:50:32/model_200.pth \
#     --classifier ../../output/cifar10/svhn_backdoored_encoder/2023-12-26-13:50:32/classifier.pkl \
#     --trigger_file /home/hrzhang/projects/badencoder_filter/optimize_filter/trigger/cifar10/2023-12-20-23-18-29/ssim0.9855_psnr30.10_lp0.0166_wd0.066_color3.966.pt \
#     --num_classes 10 \
#     > logs/cifar10-svhn.log 2>&1 &



CUDA_VISIBLE_DEVICES=0 nohup python STRIP.py \
    --dataset cifar10 --attack_mode attack \
    --encoder_usage_info stl10 \
    --encoder ../../output/stl10/cifar10_backdoored_encoder/2023-12-25-17:10:56/model_150.pth \
    --classifier ../../output/stl10/cifar10_backdoored_encoder/2023-12-25-17:10:56/classifier.pkl \
    --trigger_file /home/hrzhang/projects/badencoder_filter/output/stl10/cifar10_backdoored_encoder/2023-12-25-17:10:56/unet_filter_150_trained.pt \
    --num_classes 10 \
    > logs/stl10-cifar10.log 2>&1 &