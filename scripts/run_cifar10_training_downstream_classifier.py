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
            >./log/{encoder_usage_info}/evaluation_{key}_{encoder_usage_info}_{downstream_dataset}_f2.log 2>&1 &"


    os.system(cmd)

# _nocolorinit_with_loss0
# _color_init_no_color
# _color_loss0
# ablate_color
# randinit
# _robust1_

# run_eval(5, 'cifar10', 'stl10', 'output/cifar10/clean_encoder/model_1000.pth', 9, './trigger/cifar10/unet_filter.pt', 'truck')
# run_eval(5, 'cifar10', 'gtsrb', 'output/cifar10/clean_encoder/model_1000.pth', 12, './trigger/cifar10/unet_filter.pt', 'priority')
# run_eval(5, 'cifar10', 'svhn', 'output/cifar10/clean_encoder/model_1000.pth', 1, './trigger/cifar10/unet_filter.pt', 'one')


# run_eval(0, 'cifar10', 'stl10', 'output/cifar10/stl10_backdoored_encoder/2023-12-25-20:38:31/model_50.pth', 9, 'output/cifar10/stl10_backdoored_encoder/2023-12-25-20:38:31/unet_filter_50_trained.pt', 'truck', 'backdoor') # color + loss0
# run_eval(0, 'cifar10', 'stl10', 'output/cifar10/stl10_backdoored_encoder/2023-12-25-20:38:59/model_150.pth', 9, 'output/cifar10/stl10_backdoored_encoder/2023-12-25-20:38:59/unet_filter_150_trained.pt', 'truck', 'backdoor') # loss0

# run_eval(1, 'cifar10', 'gtsrb', './output/cifar10/gtsrb_backdoored_encoder/2024-01-28-13:17:00/model_100.pth', 12, 'output/cifar10/gtsrb_backdoored_encoder//2024-01-28-13:17:00/unet_filter_100_trained.pt', 'priority', 'backdoor') # color + loss0
# run_eval(0, 'cifar10', 'gtsrb', './output/cifar10/gtsrb_backdoored_encoder/2024-01-03-12:20:49/model_200.pth', 12, 'output/cifar10/gtsrb_backdoored_encoder/2023-12-29-13:20:56/unet_filter_200_trained.pt', 'priority', 'backdoor') # loss0

# run_eval(5, 'cifar10', 'svhn', './output/cifar10/svhn_backdoored_encoder/2023-12-26-13:50:32/model_200.pth', 1, 'output/cifar10/svhn_backdoored_encoder/2023-12-26-13:50:32/unet_filter_200_trained.pt', 'one', 'backdoor') # color + loss0
# run_eval(5, 'cifar10', 'svhn', './output/cifar10/svhn_backdoored_encoder/2024-01-05-16:59:54/model_200.pth', 1, 'output/cifar10/svhn_backdoored_encoder/2024-01-05-16:59:54/unet_filter_200_trained.pt', 'one', 'backdoor') # loss0


run_eval(1, 'stl10', 'cifar10', ' ./output/stl10/cifar10_backdoored_encoder/2024-01-31-09:39:23/model_150.pth', 0, './output/stl10/cifar10_backdoored_encoder/2024-01-31-09:39:23/unet_filter_150_trained.pt', 'airplane', 'backdoor')# color + loss0
# run_eval(5, 'stl10', 'cifar10', './output/stl10/cifar10_backdoored_encoder/2024-01-05-17:10:47/model_200.pth', 0, './output/stl10/cifar10_backdoored_encoder/2024-01-05-17:10:47/unet_filter_200_trained.pt', 'airplane', 'backdoor') # loss0

# run_eval(2, 'stl10', 'gtsrb', './output/stl10/gtsrb_backdoored_encoder/2024-01-31-09:39:19/model_100.pth', 12, './output/stl10/gtsrb_backdoored_encoder/2024-01-31-09:39:19/unet_filter_100_trained.pt', 'priority', 'backdoor')# color + loss0
# run_eval(5, 'stl10', 'gtsrb', './output/stl10/gtsrb_backdoored_encoder/2024-01-05-17:11:20/model_200.pth', 12, './output/stl10/gtsrb_backdoored_encoder/2024-01-05-17:11:20/unet_filter_200_trained.pt', 'priority', 'backdoor') # loss0

# run_eval(2, 'stl10', 'svhn', './output/stl10/svhn_backdoored_encoder/2024-01-31-09:39:14/model_50.pth', 1, './output/stl10/svhn_backdoored_encoder/2024-01-31-09:39:14/unet_filter_50_trained.pt', 'one', 'backdoor') # color + loss0
# run_eval(5, 'stl10', 'svhn', './output/stl10/svhn_backdoored_encoder/2024-01-05-17:11:23/model_200.pth', 1, './output/stl10/svhn_backdoored_encoder/2024-01-05-17:11:23/unet_filter_200_trained.pt', 'one', 'backdoor') # loss0


# run_eval(1, 'stl10', 'svhn', './output/stl10/svhn_backdoored_encoder/2023-12-16-17:34:27bpp/model_150.pth', 1, './output/stl10/svhn_backdoored_encoder/2023-12-16-17:34:27bpp/unet_filter_150_trained.pt', 'one', 'backdoor') # color + loss0
# run_eval(0, 'stl10', 'svhn', './output/stl10/svhn_backdoored_encoder/2024-01-04-20:28:10/model_200.pth', 1, './output/stl10/svhn_backdoored_encoder/2024-01-04-20:28:10/unet_filter_200_trained.pt', 'one', 'backdoor') # loss0

# run_eval(3, 'gtsrb', 'cifar10', './output/gtsrb/cifar10_backdoored_encoder/model_50.pth', 0, './trigger/gtsrb/filter.pt', 'airplane', 'backdoor')
# run_eval(4, 'gtsrb', 'stl10', './output/gtsrb/stl10_backdoored_encoder/model_50.pth', 9, './trigger/gtsrb/filter.pt', 'truck', 'backdoor')
# run_eval(3, 'gtsrb', 'svhn', './output/gtsrb/svhn_backdoored_encoder/model_50.pth', 1, './trigger/gtsrb/filter.pt', 'one', 'backdoor')



######## Wanet ########
# run_eval(0, 'cifar10', 'stl10', 'output/cifar10/stl10_backdoored_encoder/2023-12-16-20:57:00wanet/model_200.pth', 9, 'xx', 'truck', 'backdoor')
# run_eval(0, 'cifar10', 'gtsrb', './output/cifar10/gtsrb_backdoored_encoder/2023-12-16-20:58:27wanet/model_200.pth', 12, 'xx', 'priority', 'backdoor')
# run_eval(4, 'cifar10', 'svhn', './output/cifar10/svhn_backdoored_encoder/2023-12-16-20:58:48wanet/model_200.pth', 1, 'xx', 'one', 'backdoor')

# run_eval(4, 'stl10', 'cifar10', 'output/stl10/cifar10_backdoored_encoder/2023-12-16-21:02:40wanet/model_200.pth', 0, 'xx', 'airplane', 'backdoor')
# run_eval(0, 'stl10', 'gtsrb', 'output/stl10/gtsrb_backdoored_encoder/2023-12-14-20:55:35wanet/model_200.pth', 12, 'xx', 'priority', 'backdoor')
# run_eval(0, 'stl10', 'svhn', 'output/stl10/svhn_backdoored_encoder/2023-12-16-21:03:18wanet/model_200.pth', 1, 'xx', 'one', 'backdoor')
########

######## bpp ########
# run_eval(1, 'cifar10', 'stl10', 'output/cifar10/stl10_backdoored_encoder/2023-12-16-17:28:47bpp/model_200.pth', 9, 'xx', 'truck', 'backdoor')
# run_eval(1, 'cifar10', 'gtsrb', './output/cifar10/gtsrb_backdoored_encoder/2023-12-16-17:33:17bpp/model_100.pth', 12, 'xx', 'priority', 'backdoor')
# run_eval(4, 'cifar10', 'svhn', './output/cifar10/svhn_backdoored_encoder/2023-12-16-17:31:00bpp/model_100.pth', 1, 'xx', 'one', 'backdoor')

# run_eval(4, 'stl10', 'cifar10', './output/stl10/cifar10_backdoored_encoder/2023-12-16-17:34:27bpp/model_50.pth', 0, 'xx', 'airplane', 'backdoor')
# run_eval(4, 'stl10', 'gtsrb', './output/stl10/gtsrb_backdoored_encoder/2023-12-16-17:34:56bpp/model_50.pth', 12, 'xx', 'priority', 'backdoor')
# run_eval(4, 'stl10', 'svhn', './output/stl10/svhn_backdoored_encoder/2023-12-16-17:35:32bpp/model_50.pth', 1, 'xx', 'one', 'backdoor')
########

######## ctrl ########
# run_eval(2, 'cifar10', 'stl10', 'output/cifar10/stl10_backdoored_encoder/2023-12-15-16:03:27ctrl/model_200.pth', 9, 'xx', 'truck', 'backdoor')
# run_eval(2, 'cifar10', 'gtsrb', './output/cifar10/gtsrb_backdoored_encoder/2023-12-15-16:03:23ctrl/model_50.pth', 12, 'xx', 'priority', 'backdoor')
# run_eval(3, 'cifar10', 'svhn', './output/cifar10/svhn_backdoored_encoder/2023-12-15-16:03:20ctrl/model_200.pth', 1, 'xx', 'one', 'backdoor')


########
# ins - kelvin

# run_eval(4, 'stl10', 'cifar10', './output/stl10/cifar10_backdoored_encoder/2023-12-29-21:21:21/model_200.pth', 0, 'xx', 'airplane', 'backdoor')
# run_eval(4, 'stl10', 'gtsrb', './output/stl10/gtsrb_backdoored_encoder/2023-12-29-21:21:16/model_200.pth', 12, 'xx', 'priority', 'backdoor')
# run_eval(4, 'stl10', 'svhn', './output/stl10/svhn_backdoored_encoder/2023-12-29-21:21:12/model_200.pth', 1, 'xx', 'one', 'backdoor')


########
# ins - xpro2
# run_eval(0, 'cifar10', 'stl10', 'output/cifar10/stl10_backdoored_encoder/2023-12-30-00:04:40/model_200.pth', 9, 'xx', 'truck', 'backdoor')
# run_eval(0, 'cifar10', 'gtsrb', './output/cifar10/gtsrb_backdoored_encoder/2023-12-30-00:05:56/model_200.pth', 12, 'xx', 'priority', 'backdoor')
# run_eval(0, 'cifar10', 'svhn', './output/cifar10/svhn_backdoored1_encoder/2023-12-30-00:05:27/model_200.pth', 1, 'xx', 'one', 'backdoor')

# run_eval(4, 'stl10', 'cifar10', './output/stl10/cifar10_backdoored_encoder/2023-12-30-00:02:09/model_200.pth', 0, 'xx', 'airplane', 'backdoor')
# run_eval(4, 'stl10', 'gtsrb', './output/stl10/gtsrb_backdoored_encoder/2023-12-30-00:02:53/model_200.pth', 12, 'xx', 'priority', 'backdoor')
# run_eval(4, 'stl10', 'svhn', './output/stl10/svhn_backdoored_encoder/2023-12-30-00:03:45/model_200.pth', 1, 'xx', 'one', 'backdoor')


##### patch backdoor
# run_eval(3, 'cifar10', 'stl10', './output/cifar10/stl10_backdoored_encoder/model_200.pth', 9, './trigger/trigger_pt_white_21_10_ap_replace.npz', 'truck', 'backdoor')
# run_eval(4, 'cifar10', 'gtsrb', './output/cifar10/gtsrb_backdoored_encoder/model_200.pth', 12, './trigger/trigger_pt_white_21_10_ap_replace.npz', 'priority', 'backdoor')
# run_eval(5, 'cifar10', 'svhn', './output/cifar10/svhn_backdoored_encoder/model_200.pth', 1, './trigger/trigger_pt_white_21_10_ap_replace.npz', 'one', 'backdoor')

# run_eval(1, 'stl10', 'cifar10', './output/stl10/cifar10_backdoored_encoder/model_200.pth', 0, './trigger/trigger_pt_white_21_10_ap_replace.npz', 'airplane', 'backdoor')
# run_eval(1, 'stl10', 'gtsrb', './output/stl10/gtsrb_backdoored_encoder/model_200.pth', 14, './trigger/trigger_pt_white_21_10_ap_replace.npz', 'stop', 'backdoor')
# run_eval(1, 'stl10', 'svhn', './output/stl10/svhn_backdoored_encoder/model_200.pth', 1, './trigger/trigger_pt_white_21_10_ap_replace.npz', 'one', 'backdoor')# robustness
