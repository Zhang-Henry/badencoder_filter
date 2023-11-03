# CUDA_VISIBLE_DEVICES=1 nohup python optimize_filter.py > log/cifar10/filter_sameweight_$(date +"%Y-%m-%d-%H-%M-%S").log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python optimize_filter_stl.py > log/stl10/filter_sameweight.log

# CUDA_VISIBLE_DEVICES=0 nohup python optimize_filter.py > log/cifar10/filter_$(date +"%Y-%m-%d-%H-%M-%S")_cifar10_ablation.log 2>&1 &
# CUDA_VISIBLE_DEVICES=4 nohup python optimize_filter.py > log/filter_$(date +"%Y-%m-%d-%H-%M-%S")_imagenet.log 2>&1 &
# CUDA_VISIBLE_DEVICES=5 nohup python optimize_filter.py > log/filter_$(date +"%Y-%m-%d-%H-%M-%S")_cifar_ab.log 2>&1 &

# CUDA_VISIBLE_DEVICES=2 nohup python optimize_filter_imagenet.py > log/imagenet/filter_unet_$(date +"%Y-%m-%d-%H-%M-%S").log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup python optimize_filter_imagenet.py > log/imagenet/filter_unet.log


CUDA_VISIBLE_DEVICES=2 nohup python optimize_filter_imagenet_unet.py > log/imagenet/filter_unet_$(date +"%Y-%m-%d-%H-%M-%S").log 2>&1 &
# CUDA_VISIBLE_DEVICES=4 nohup python optimize_filter_imagenet_unet.py > log/imagenet/filter_unet.log 2>&1 &