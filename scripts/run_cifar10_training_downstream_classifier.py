import os

# if not os.path.exists('./log/cifar10'):
#     os.makedirs('./log/cifar10')

print('Start evaluation')
def run_eval(gpu, encoder_usage_info, downstream_dataset, encoder, reference_label, trigger, reference_file, key='clean'):
    cmd = f"CUDA_VISIBLE_DEVICES=1 nohup python3 -u training_downstream_classifier.py \
            --dataset {downstream_dataset} \
            --trigger_file {trigger} \
            --encoder {encoder} \
            --encoder_usage_info {encoder_usage_info} \
            --reference_label {reference_label} \
            --reference_file ./reference/{encoder_usage_info}/{reference_file}.npz \
            --gpu {gpu} \
            --seed {3407} \
            >./log/{encoder_usage_info}/evaluation_{key}_{encoder_usage_info}_{downstream_dataset}.txt 2>&1 &"

    os.system(cmd)



# run_eval(5, 'cifar10', 'stl10', 'output/cifar10/clean_encoder/model_1000.pth', 9, './trigger/trigger_pt_white_21_10_ap_replace.npz', 'truck')
# run_eval(5, 'cifar10', 'gtsrb', 'output/cifar10/clean_encoder/model_1000.pth', 12, './trigger/trigger_pt_white_21_10_ap_replace.npz', 'priority')
# run_eval(5, 'cifar10', 'svhn', 'output/cifar10/clean_encoder/model_1000.pth', 1, './trigger/trigger_pt_white_21_10_ap_replace.npz', 'one')


# run_eval(1, 'cifar10', 'stl10', './output/cifar10/stl10_backdoored_encoder/model_200.pth', 9, './trigger/trigger_pt_white_21_10_ap_replace.npz', 'truck', 'backdoor')
# run_eval(1, 'cifar10', 'gtsrb', './output/cifar10/gtsrb_backdoored_encoder/model_200.pth', 12, './trigger/trigger_pt_white_21_10_ap_replace.npz', 'priority', 'backdoor')
# run_eval(1, 'cifar10', 'svhn', './output/cifar10/svhn_backdoored_encoder/model_200.pth', 1, './trigger/trigger_pt_white_21_10_ap_replace.npz', 'one', 'backdoor')


# run_eval(1, 'stl10', 'cifar10', './output/stl10/cifar10_backdoored_encoder/model_200.pth', 0, './trigger/trigger_pt_white_21_10_ap_replace.npz', 'airplane', 'backdoor')
# run_eval(1, 'stl10', 'gtsrb', './output/stl10/gtsrb_backdoored_encoder/model_200.pth', 14, './trigger/trigger_pt_white_21_10_ap_replace.npz', 'stop', 'backdoor')
# run_eval(1, 'stl10', 'gtsrb', './output/stl10/gtsrb_backdoored_encoder/model_200.pth', 12, './trigger/trigger_pt_white_21_10_ap_replace.npz', 'priority', 'backdoor')
run_eval(1, 'stl10', 'svhn', './output/stl10/svhn_backdoored_encoder/model_200.pth', 1, './trigger/trigger_pt_white_21_10_ap_replace.npz', 'one', 'backdoor')