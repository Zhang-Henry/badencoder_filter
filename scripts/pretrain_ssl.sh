## moco_v2
# nohup python -u pretrain_ssl/moco_memory_bank.py > log/clean_encoder/moco_cifar.log 2>&1 &

## simsiam
CUDA_VISIBLE_DEVICES=3 nohup python -u pretrain_ssl/cifar10_benchmark.py > log/clean_encoder/DINOModel_NNCLRModel_cifar.log 2>&1 &

