import os

if not os.path.exists('./log/clip/'):
    os.makedirs('./log/clip/')

def evaluate_clip_finetune(gpu, encoder_usage_info, downstream_dataset, encoder, reference_label, trigger, reference):
    os.environ["CUDA_VISIBLE_DEVICES"]= str(gpu)

    cmd = f"nohup python3 -u training_downstream_classifier.py \
            --encoder_usage_info {encoder_usage_info} \
            --dataset {downstream_dataset} \
            --trigger_file {trigger} \
            --encoder {encoder} \
            --reference_label {reference_label} \
            --reference_file ./reference/CLIP/{reference}.npz \
            --gpu {gpu} \
            >./log/clip/evaluation_backdoor_{downstream_dataset}.txt &"

    os.system(cmd)


def eval_clean_model(gpu, encoder_usage_info, downstream_dataset, encoder, reference_label, trigger='./trigger/trigger_pt_white_173_50_ap_replace.npz', reference=None):
    os.environ["CUDA_VISIBLE_DEVICES"]= str(gpu)

    cmd = f"nohup python3 -u training_downstream_classifier.py \
            --encoder_usage_info {encoder_usage_info} \
            --dataset {downstream_dataset} \
            --trigger_file {trigger} \
            --encoder {encoder} \
            --reference_label {reference_label} \
            --reference_file ./reference/CLIP/{reference}.npz \
            --gpu {gpu} \
            >./log/clip/eval_clean_{downstream_dataset}.txt 2>&1 &"

    os.system(cmd)


# evaluate_clip_finetune(6, 'CLIP', 'gtsrb', './output/CLIP/backdoor/priority/model_200.pth', 12, './trigger/trigger_pt_white_173_50_ap_replace.npz', 'priority')
evaluate_clip_finetune(1, 'CLIP', 'stl10', './output/CLIP/backdoor/truck/model_200.pth', 9, './trigger/trigger_pt_white_173_50_ap_replace.npz', 'truck')
# evaluate_clip_finetune(4, 'CLIP', 'svhn', 'output/CLIP/svhn_backdoored_encoder/2024-05-13-00:01:32/model_25.pth', 1, 'output/CLIP/svhn_backdoored_encoder/2024-05-13-00:01:32/unet_filter_25_trained.pt', 'one')

# eval_clean_model(1, 'CLIP', 'gtsrb', './output/CLIP/clean_encoder/encode_image.pth', 12, reference='priority')
# eval_clean_model(3, 'CLIP', 'stl10', './output/CLIP/clean_encoder/encode_image.pth', 9, reference='truck')
# eval_clean_model(4, 'CLIP', 'svhn', './output/CLIP/clean_encoder/encode_image.pth', 1, reference='one')
