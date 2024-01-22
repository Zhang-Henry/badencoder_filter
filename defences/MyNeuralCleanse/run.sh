# nohup python backdoor.py  > backdoor.log
# nohup python train.py  > train.log
nohup python -u train2.py --dataset cifar10 \
 --encoder_usage_info stl10 \
 --encoder ../../output/stl10/svhn_backdoored_encoder/2023-12-28-17:03:53/model_50.pth \
 --classifier ../../output/stl10/svhn_backdoored_encoder/2023-12-28-17:03:53/classifier.pkl \
 --num_classes 10 > logs/stl10-cifar10.log 2>&1 &


nohup python -u train2.py --dataset cifar10 \
 --encoder_usage_info stl10 \
 --encoder ../../output/stl10/svhn_backdoored_encoder/2023-12-28-17:03:53/model_50.pth \
 --classifier ../../output/stl10/svhn_backdoored_encoder/2023-12-28-17:03:53/classifier.pkl \
 --num_classes 10 > logs/stl10-cifar10.log 2>&1 &