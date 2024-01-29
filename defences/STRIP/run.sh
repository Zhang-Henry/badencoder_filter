############ stl10 ############

# CUDA_VISIBLE_DEVICES=0 nohup python STRIP.py \
#     --dataset cifar10 --attack_mode attack \
#     --encoder_usage_info stl10 \
#     --encoder ../../output/stl10/cifar10_backdoored_encoder/2023-12-25-17:10:56/model_150.pth \
#     --classifier ../../output/stl10/cifar10_backdoored_encoder/2023-12-25-17:10:56/classifier.pkl \
#     --trigger_file /home/hrzhang/projects/badencoder_filter/output/stl10/cifar10_backdoored_encoder/2023-12-25-17:10:56/unet_filter_150_trained.pt \
#     --num_classes 10 \
#     > logs/stl10-cifar10.log 2>&1 &


# CUDA_VISIBLE_DEVICES=0 nohup python STRIP.py \
#     --dataset svhn --attack_mode attack \
#     --encoder_usage_info stl10 \
#     --encoder ../../output/stl10/svhn_backdoored_encoder/2023-12-28-17:03:53/model_50.pth \
#     --classifier ../../output/stl10/svhn_backdoored_encoder/2023-12-28-17:03:53/classifier.pkl \
#     --trigger_file /home/hrzhang/projects/badencoder_filter/output/stl10/svhn_backdoored_encoder/2023-12-28-17:03:53/unet_filter_50_trained.pt \
#     --num_classes 10 \
#     > logs/stl10-svhn.log 2>&1 &


# CUDA_VISIBLE_DEVICES=0 nohup python STRIP.py \
#     --dataset gtsrb --attack_mode attack \
#     --encoder_usage_info stl10 \
#     --encoder ../../output/stl10/gtsrb_backdoored_encoder/2024-01-04-20:26:45/model_200.pth \
#     --classifier ../../output/stl10/gtsrb_backdoored_encoder/2024-01-04-20:26:45/classifier.pkl \
#     --trigger_file /home/hrzhang/projects/badencoder_filter/output/stl10/gtsrb_backdoored_encoder/2024-01-04-20:26:45/unet_filter_200_trained.pt \
#     --num_classes 43 \
#     > logs/stl10-gtsrb.log 2>&1 &



############ cifar10 ############
# CUDA_VISIBLE_DEVICES=0 nohup python STRIP.py \
#     --dataset stl10 --attack_mode attack \
#     --encoder_usage_info cifar10 \
#     --encoder ../../output/cifar10/stl10_backdoored_encoder/2023-12-25-20:38:31/model_50.pth \
#     --classifier ../../output/cifar10/stl10_backdoored_encoder/2023-12-25-20:38:31/classifier.pkl \
#     --trigger_file /home/hrzhang/projects/badencoder_filter/output/cifar10/stl10_backdoored_encoder/2023-12-25-20:38:31/unet_filter_50_trained.pt \
#     --num_classes 10 \
#     > logs/cifar10-stl10.log 2>&1 &


# CUDA_VISIBLE_DEVICES=0 nohup python STRIP.py \
#     --dataset gtsrb --attack_mode attack \
#     --encoder_usage_info cifar10 \
#     --encoder ../../output/cifar10/gtsrb_backdoored_encoder/2023-12-20-16:03:49/model_100.pth \
#     --classifier ../../output/cifar10/gtsrb_backdoored_encoder/2023-12-20-16:03:49/classifier.pkl \
#     --trigger_file /home/hrzhang/projects/badencoder_filter/output/cifar10/gtsrb_backdoored_encoder/2023-12-20-16:03:49/unet_filter_100_trained.pt \
#     --num_classes 43 \
#     > logs/cifar10-gtsrb.log 2>&1 &

# CUDA_VISIBLE_DEVICES=1 nohup python STRIP.py \
#     --dataset gtsrb --attack_mode attack \
#     --encoder_usage_info cifar10 \
#     --encoder ../../output/cifar10/gtsrb_backdoored_encoder/2024-01-03-12:21:01/model_200.pth \
#     --classifier ../../output/cifar10/gtsrb_backdoored_encoder/2024-01-03-12:21:01/classifier.pkl \
#     --trigger_file /home/hrzhang/projects/badencoder_filter/output/cifar10/gtsrb_backdoored_encoder/2024-01-03-12:21:01/unet_filter_200_trained.pt \
#     --num_classes 43 \
#     > logs/cifar10-gtsrb.log 2>&1 &

# CUDA_VISIBLE_DEVICES=1 nohup python STRIP.py \
#     --dataset gtsrb --attack_mode no_attack \
#     --encoder_usage_info cifar10 \
#     --encoder ../../output/cifar10/gtsrb_backdoored_encoder/2024-01-03-12:21:01/model_25.pth \
#     --classifier ../../output/cifar10/gtsrb_backdoored_encoder/2024-01-03-12:21:01/classifier.pkl \
#     --trigger_file /home/hrzhang/projects/badencoder_filter/output/cifar10/gtsrb_backdoored_encoder/2024-01-03-12:21:01/unet_filter_25_trained.pt \
#     --num_classes 43 \
#     > logs/cifar10-gtsrb.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python -u STRIP.py \
    --dataset gtsrb --attack_mode attack \
    --encoder_usage_info cifar10 \
    --encoder ../../output/cifar10/gtsrb_backdoored_encoder/2024-01-28-13:17:00/model_100.pth \
    --classifier ../../output/cifar10/gtsrb_backdoored_encoder/2024-01-28-13:17:00/classifier.pkl \
    --trigger_file /home/hrzhang/projects/badencoder_filter/output/cifar10/gtsrb_backdoored_encoder/2024-01-28-13:17:00/unet_filter_100_trained.pt \
    --num_classes 43 \
    > logs/cifar10-gtsrb.log 2>&1 &

# CUDA_VISIBLE_DEVICES=0 nohup python STRIP.py \
#     --dataset svhn --attack_mode attack \
#     --encoder_usage_info cifar10 \
#     --encoder ../../output/cifar10/svhn_backdoored_encoder/2023-12-26-13:50:32/model_200.pth \
#     --classifier ../../output/cifar10/svhn_backdoored_encoder/2023-12-26-13:50:32/classifier.pkl \
#     --trigger_file /home/hrzhang/projects/badencoder_filter/output/cifar10/svhn_backdoored_encoder/2023-12-26-13:50:32/unet_filter_200_trained.pt \
#     --num_classes 10 \
#     > logs/cifar10-svhn.log 2>&1 &