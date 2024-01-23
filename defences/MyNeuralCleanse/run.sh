# nohup python backdoor.py  > backdoor.log
# nohup python train.py  > train.log
## stl-cifar
# nohup python -u train2.py --dataset cifar10 \
#  --encoder_usage_info stl10 \
#  --encoder ../../output/stl10/svhn_backdoored_encoder/2023-12-28-17:03:53/model_50.pth \
#  --classifier ../../output/stl10/svhn_backdoored_encoder/2023-12-28-17:03:53/classifier.pkl \
#  --num_classes 10 > logs/stl10-cifar10.log 2>&1 &

## stl-gtsrb
# nohup python -u train2.py --dataset gtsrb \
#  --encoder_usage_info stl10 \
#  --encoder ../../output/stl10/gtsrb_backdoored_encoder/2023-12-25-16:45:06/model_100.pth \
#  --classifier ../../output/stl10/gtsrb_backdoored_encoder/2023-12-25-16:45:06/classifier.pkl \
#  --num_classes 43 > logs/stl10-gtsrb.log 2>&1 &

# stl-svhn
# nohup python -u train2.py --dataset svhn \
#  --encoder_usage_info stl10 \
#  --encoder ../../output/stl10/svhn_backdoored_encoder/2023-12-28-17:03:53/model_50.pth \
#  --classifier ../../output/stl10/svhn_backdoored_encoder/2023-12-28-17:03:53/classifier.pkl \
#  --num_classes 10 > logs/stl10-svhn.log 2>&1 &


## cifar-stl
# nohup python -u train2.py --dataset stl10 \
#  --encoder_usage_info cifar10 \
#  --encoder ../../output/cifar10/stl10_backdoored_encoder/2023-12-25-20:38:31/model_50.pth \
#  --classifier ../../output/cifar10/stl10_backdoored_encoder/2023-12-25-20:38:31/classifier.pkl \
#  --num_classes 10 > logs/cifar10-stl10.log 2>&1 &

## cifar-gtsrb
# nohup python -u train2.py --dataset gtsrb \
#  --encoder_usage_info cifar10 \
#  --encoder ../../output/cifar10/gtsrb_backdoored_encoder/2023-12-20-16:03:49/model_100.pth \
#  --classifier ../../output/cifar10/gtsrb_backdoored_encoder/2023-12-20-16:03:49/classifier.pkl \
#  --num_classes 43 > logs/cifar10-gtsrb.log 2>&1 &

# cifar-svhn
nohup python -u train2.py --dataset svhn \
 --encoder_usage_info cifar10 \
 --encoder ../../output/cifar10/svhn_backdoored_encoder/2023-12-26-13:50:32/model_200.pth \
 --classifier ../../output/cifar10/svhn_backdoored_encoder/2023-12-26-13:50:32/classifier.pkl \
 --num_classes 10 > logs/cifar10-svhn.log 2>&1 &