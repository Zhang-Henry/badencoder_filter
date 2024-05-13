python3 -u eval_beatrix.py \
    --batch_size 256 \
    --shadow_dataset cifar10 \
    --pretrained_encoder {model-path} \
    --encoder_usage_info cifar10 \
    --gpu 0 \
    --trigger_file ./trigger/trigger_pt_white_21_10_ap_replace.npz --lr 0.1 --epochs 60 \
    --reference_file ./reference/gtsrb_l12_n3.npz \
    --cut_threshold 8000 