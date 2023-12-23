import os

# if not os.path.exists('./log/cifar10'):
#     os.makedirs('./log/cifar10')

print('Start evaluation')
def run_eval(gpu, encoder_usage_info, downstream_dataset, encoder, reference_label, trigger, reference_file, key='clean'):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    cmd = f"nohup python3 -u training_downstream_classifier.py \
            --dataset {downstream_dataset} \
            --trigger_file {trigger} \
            --encoder {encoder} \
            --encoder_usage_info {encoder_usage_info} \
            --reference_label {reference_label} \
            --reference_file ./reference/{encoder_usage_info}/{reference_file}.npz \
            --gpu {gpu} \
            >./log/{encoder_usage_info}/evaluation_{key}_{encoder_usage_info}_{downstream_dataset}_ablate_color2.log 2>&1 &"

    os.system(cmd)

# _nocolorinit_with_loss0
# _color_init_no_color
# _color_loss0
# ablate_color

# run_eval(5, 'cifar10', 'stl10', 'output/cifar10/clean_encoder/model_1000.pth', 9, './trigger/cifar10/unet_filter.pt', 'truck')
# run_eval(5, 'cifar10', 'gtsrb', 'output/cifar10/clean_encoder/model_1000.pth', 12, './trigger/cifar10/unet_filter.pt', 'priority')
# run_eval(5, 'cifar10', 'svhn', 'output/cifar10/clean_encoder/model_1000.pth', 1, './trigger/cifar10/unet_filter.pt', 'one')


# run_eval(4, 'cifar10', 'stl10', 'output/cifar10/stl10_backdoored_encoder/2023-12-22-11:16:37/model_200.pth', 9, 'output/cifar10/stl10_backdoored_encoder/2023-12-22-11:16:37/unet_filter_200_trained.pt', 'truck', 'backdoor')
# run_eval(0, 'cifar10', 'gtsrb', './output/cifar10/gtsrb_backdoored_encoder/2023-12-20-16:03:49/model_75.pth', 12, 'output/cifar10/gtsrb_backdoored_encoder/2023-12-20-16:03:49/unet_filter_75_trained.pt', 'priority', 'backdoor')
# run_eval(0, 'cifar10', 'svhn', './output/cifar10/svhn_backdoored_encoder/2023-12-20-16:04:09/model_75.pth', 1, 'output/cifar10/svhn_backdoored_encoder/2023-12-20-16:04:09/unet_filter_75_trained.pt', 'one', 'backdoor')
# run_eval(0, 'cifar10', 'svhn', './output/cifar10/svhn_backdoored_encoder/2023-12-21-00:04:42/model_75.pth', 1, 'output/cifar10/svhn_backdoored_encoder/2023-12-21-00:04:42/unet_filter_75_trained.pt', 'one', 'backdoor')




# run_eval(0, 'stl10', 'cifar10', './output/stl10/cifar10_backdoored_encoder/2023-12-22-14:12:40/model_50.pth', 0, './output/stl10/cifar10_backdoored_encoder/2023-12-22-14:12:40/unet_filter_50_trained.pt', 'airplane', 'backdoor')
# run_eval(0, 'stl10', 'gtsrb', './output/stl10/gtsrb_backdoored_encoder/2023-12-19-14:44:29/model_50.pth', 12, './output/stl10/gtsrb_backdoored_encoder/2023-12-19-14:44:29/unet_filter_50_trained.pt', 'priority', 'backdoor')
# run_eval(4, 'stl10', 'svhn', './output/stl10/svhn_backdoored_encoder/2023-12-17-13:22:59/model_50.pth', 1, './output/stl10/svhn_backdoored_encoder/2023-12-17-13:22:59/unet_filter_50_trained.pt', 'one', 'backdoor')



# run_eval(3, 'gtsrb', 'cifar10', './output/gtsrb/cifar10_backdoored_encoder/model_50.pth', 0, './trigger/gtsrb/filter.pt', 'airplane', 'backdoor')
# run_eval(4, 'gtsrb', 'stl10', './output/gtsrb/stl10_backdoored_encoder/model_50.pth', 9, './trigger/gtsrb/filter.pt', 'truck', 'backdoor')
# run_eval(3, 'gtsrb', 'svhn', './output/gtsrb/svhn_backdoored_encoder/model_50.pth', 1, './trigger/gtsrb/filter.pt', 'one', 'backdoor')



######## Wanet ########
# run_eval(0, 'cifar10', 'stl10', 'output/cifar10/stl10_backdoored_encoder/2023-12-16-20:57:00wanet/model_200.pth', 9, 'xx', 'truck', 'backdoor')
# run_eval(0, 'cifar10', 'gtsrb', './output/cifar10/gtsrb_backdoored_encoder/2023-12-16-20:58:27wanet/model_200.pth', 12, 'xx', 'priority', 'backdoor')
# run_eval(4, 'cifar10', 'svhn', './output/cifar10/svhn_backdoored_encoder/2023-12-16-20:58:48wanet/model_200.pth', 1, 'xx', 'one', 'backdoor')

# run_eval(4, 'stl10', 'cifar10', './output/stl10/cifar10_backdoored_encoder/2023-12-11-12:21:05/model_200.pth', 0, 'xx', 'airplane', 'backdoor')
# run_eval(1, 'stl10', 'gtsrb', './output/stl10/gtsrb_backdoored_encoder/2023-12-11-12:21:14/model_200.pth', 12, 'xx', 'priority', 'backdoor')
# run_eval(1, 'stl10', 'svhn', './output/stl10/svhn_backdoored_encoder/2023-12-11-12:21:19/model_200.pth', 1, 'xx', 'one', 'backdoor')
########

######## bpp ########
# run_eval(1, 'cifar10', 'stl10', 'output/cifar10/stl10_backdoored_encoder/2023-12-16-17:28:47bpp/model_200.pth', 9, 'xx', 'truck', 'backdoor')
# run_eval(0, 'cifar10', 'gtsrb', './output/cifar10/gtsrb_backdoored_encoder/2023-12-16-17:33:17bpp/model_200.pth', 12, 'xx', 'priority', 'backdoor')
# run_eval(4, 'cifar10', 'svhn', './output/cifar10/svhn_backdoored_encoder/2023-12-16-17:31:00bpp/model_200.pth', 1, 'xx', 'one', 'backdoor')

# run_eval(4, 'stl10', 'cifar10', './output/stl10/cifar10_backdoored_encoder/2023-12-16-17:34:27bpp/model_175.pth', 0, 'xx', 'airplane', 'backdoor')
# run_eval(4, 'stl10', 'gtsrb', './output/stl10/gtsrb_backdoored_encoder/2023-12-16-17:34:56bpp/model_175.pth', 12, 'xx', 'priority', 'backdoor')
# run_eval(4, 'stl10', 'svhn', './output/stl10/svhn_backdoored_encoder/2023-12-16-17:35:32bpp/model_175.pth', 1, 'xx', 'one', 'backdoor')
########

######## ctrl ########
# run_eval(2, 'cifar10', 'stl10', 'output/cifar10/stl10_backdoored_encoder/2023-12-15-16:03:27ctrl/model_200.pth', 9, 'xx', 'truck', 'backdoor')
# run_eval(2, 'cifar10', 'gtsrb', './output/cifar10/gtsrb_backdoored_encoder/2023-12-15-16:03:23ctrl/model_200.pth', 12, 'xx', 'priority', 'backdoor')
# run_eval(3, 'cifar10', 'svhn', './output/cifar10/svhn_backdoored_encoder/2023-12-15-16:03:20ctrl/model_200.pth', 1, 'xx', 'one', 'backdoor')


########