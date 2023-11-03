######### Finetune #############
timestamp=$(date +"%Y-%m-%d-%H-%M-%S")

nohup python main.py \
    --timestamp $timestamp \
    --lr 0.01 \
    --gpu 2 \
    --batch_size 256 \
    --n_epoch 100 \
    --step_size 60 \
    --mode finetune_backbone \
    > backbone/logs/simclr_finetune_all_nofreeze_$timestamp.log 2>&1 &