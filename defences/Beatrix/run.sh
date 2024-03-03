nohup python -u train.py --dataset cifar10 --target_label 1 --gpu 4 > logs/cifar10_1.log 2>&1 &
## python train.py --dataset gtsrb --target_label 2 --gpu 0