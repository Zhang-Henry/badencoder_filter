import os

# if not os.path.exists('./log/cifar10'):
#     os.makedirs('./log/cifar10')

print('Start evaluation')
def run_eval(gpu, encoder_usage_info, downstream_dataset, encoder, reference_label, trigger, reference_file, noise,key='clean'):
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
            --noise {noise} \
            >./log/{encoder_usage_info}/robust/evaluation_{key}_{encoder_usage_info}_{downstream_dataset}_robust_JPEGcompression9.log 2>&1 &"


    os.system(cmd)

# _robust_{noise}2
# _robust_JPEGcompression2_
# _robust_salt_and_pepper_noise0.08
# _robust_poisson_noise10


## CIFAR10 - STL10
# run_eval(0, 'cifar10', 'stl10', './output/cifar10/stl10_backdoored_encoder/2024-01-05-17:08:16/model_200.pth', 9, './output/cifar10/stl10_backdoored_encoder/2024-01-05-17:08:16/unet_filter_200_trained.pt', 'truck', 'GaussianBlur', 'backdoor')# color + loss0
# run_eval(0, 'cifar10', 'stl10', './output/cifar10/stl10_backdoored_encoder/2024-01-05-17:10:47/model_200.pth', 9, './output/cifar10/stl10_backdoored_encoder/2024-01-05-17:10:47/unet_filter_200_trained.pt', 'truck', 'GaussianBlur', 'backdoor') # loss0

# run_eval(2, 'cifar10', 'stl10', './output/cifar10/stl10_backdoored_encoder/2023-12-25-20:38:31/model_150.pth', 9, './output/cifar10/stl10_backdoored_encoder/2023-12-25-20:38:31/unet_filter_150_trained.pt', 'truck', 'salt_and_pepper_noise', 'backdoor') # loss0 + loss0
# run_eval(2, 'cifar10', 'stl10', 'output/cifar10/stl10_backdoored_encoder/2023-12-25-20:38:59/model_150.pth', 9, './output/cifar10/stl10_backdoored_encoder/2023-12-25-20:38:59/unet_filter_150_trained.pt', 'truck', 'salt_and_pepper_noise', 'backdoor')# color

# run_eval(0, 'cifar10', 'stl10', 'output/cifar10/stl10_backdoored_encoder/2023-12-25-20:38:31/model_150.pth', 9, './output/cifar10/stl10_backdoored_encoder/2023-12-25-20:38:31/unet_filter_150_trained.pt', 'truck', 'JPEGcompression','backdoor')# color + loss0
# run_eval(0, 'cifar10', 'stl10', 'output/cifar10/stl10_backdoored_encoder/2023-12-25-20:38:59/model_150.pth', 9, './output/cifar10/stl10_backdoored_encoder/2023-12-25-20:38:59/unet_filter_150_trained.pt', 'truck', 'JPEGcompression','backdoor') # loss0

# run_eval(0, 'cifar10', 'stl10', 'output/cifar10/stl10_backdoored_encoder/2023-12-25-20:38:31/model_150.pth', 9, './output/cifar10/stl10_backdoored_encoder/2023-12-25-20:38:31/unet_filter_150_trained.pt', 'truck', 'None','backdoor') # color + loss0
# run_eval(0, 'cifar10', 'stl10', 'output/cifar10/stl10_backdoored_encoder/2023-12-25-20:38:59/model_150.pth', 9, './output/cifar10/stl10_backdoored_encoder/2023-12-25-20:38:59/unet_filter_150_trained.pt', 'truck', 'None','backdoor')# loss0

# run_eval(0, 'cifar10', 'stl10', './output/cifar10/stl10_backdoored_encoder/2023-12-25-20:38:31/model_150.pth', 9, './output/cifar10/stl10_backdoored_encoder/2023-12-25-20:38:31/unet_filter_150_trained.pt', 'truck', 'poisson_noise', 'backdoor')# color + loss0
# run_eval(0, 'cifar10', 'stl10', './output/cifar10/stl10_backdoored_encoder/2023-12-25-20:38:59/model_150.pth', 9, './output/cifar10/stl10_backdoored_encoder/2023-12-25-20:38:59/unet_filter_150_trained.pt', 'truck', 'poisson_noise', 'backdoor')

### GTSRB

# run_eval(3, 'cifar10', 'gtsrb', 'output/cifar10/gtsrb_backdoored_encoder/2024-01-03-12:21:01/model_200.pth', 12, './output/cifar10/gtsrb_backdoored_encoder/2024-01-03-12:21:01/unet_filter_200_trained.pt', 'priority', 'salt_and_pepper_noise', 'backdoor') # loss0 + loss0
# run_eval(4, 'cifar10', 'gtsrb', 'output/cifar10/gtsrb_backdoored_encoder/2024-01-03-12:20:49/model_200.pth', 12, './output/cifar10/gtsrb_backdoored_encoder/2024-01-03-12:20:49/unet_filter_200_trained.pt', 'priority', 'salt_and_pepper_noise', 'backdoor')# color

run_eval(2, 'cifar10', 'gtsrb', 'output/cifar10/gtsrb_backdoored_encoder/2024-01-03-12:21:01/model_200.pth', 12, './output/cifar10/gtsrb_backdoored_encoder/2024-01-03-12:21:01/unet_filter_200_trained.pt', 'priority', 'JPEGcompression','backdoor')# color + loss0
# run_eval(1, 'cifar10', 'gtsrb', 'output/cifar10/gtsrb_backdoored_encoder/2024-01-03-12:20:49/model_200.pth', 12, './output/cifar10/gtsrb_backdoored_encoder/2024-01-03-12:20:49/unet_filter_200_trained.pt', 'priority', 'JPEGcompression','backdoor') # loss0


# run_eval(2, 'cifar10', 'gtsrb', 'output/cifar10/gtsrb_backdoored_encoder/2024-01-03-12:21:01/model_200.pth', 12, './output/cifar10/gtsrb_backdoored_encoder/2024-01-03-12:21:01/unet_filter_200_trained.pt', 'priority', 'poisson_noise', 'backdoor')# color + loss0
# run_eval(2, 'cifar10', 'gtsrb', './output/cifar10/gtsrb_backdoored_encoder/2024-01-03-12:20:49/model_200.pth', 12, './output/cifar10/gtsrb_backdoored_encoder/2024-01-03-12:20:49/unet_filter_200_trained.pt', 'priority', 'poisson_noise', 'backdoor')

# run_eval(5, 'cifar10', 'gtsrb', 'output/cifar10/gtsrb_backdoored_encoder/2024-01-03-12:21:01/model_200.pth', 12, './output/cifar10/gtsrb_backdoored_encoder/2024-01-03-12:21:01/unet_filter_200_trained.pt', 'priority', 'none', 'backdoor')# color + loss0
# run_eval(5, 'cifar10', 'gtsrb', './output/cifar10/gtsrb_backdoored_encoder/2024-01-03-12:20:49/model_200.pth', 12, './output/cifar10/gtsrb_backdoored_encoder/2024-01-03-12:20:49/unet_filter_200_trained.pt', 'priority', '', 'backdoor')
########################## STL10 - CIFAR10

# run_eval(0, 'stl10', 'cifar10', './output/stl10/cifar10_backdoored_encoder/2024-01-05-17:08:16/model_200.pth', 0, './output/stl10/cifar10_backdoored_encoder/2024-01-05-17:08:16/unet_filter_200_trained.pt', 'airplane', 'GaussianBlur', 'backdoor')# color + loss0
# run_eval(0, 'stl10', 'cifar10', './output/stl10/cifar10_backdoored_encoder/2024-01-05-17:10:47/model_200.pth', 0, './output/stl10/cifar10_backdoored_encoder/2024-01-05-17:10:47/unet_filter_200_trained.pt', 'airplane', 'GaussianBlur', 'backdoor') # loss0

# run_eval(0, 'stl10', 'cifar10', './output/stl10/cifar10_backdoored_encoder/2024-01-05-17:08:16/model_200.pth', 0, './output/stl10/cifar10_backdoored_encoder/2024-01-05-17:08:16/unet_filter_200_trained.pt', 'airplane', 'salt_and_pepper_noise', 'backdoor')# color + loss0
# run_eval(0, 'stl10', 'cifar10', './output/stl10/cifar10_backdoored_encoder/2024-01-05-17:10:47/model_200.pth', 0, './output/stl10/cifar10_backdoored_encoder/2024-01-05-17:10:47/unet_filter_200_trained.pt', 'airplane', 'salt_and_pepper_noise', 'backdoor') # loss0

# run_eval(0, 'stl10', 'cifar10', './output/stl10/cifar10_backdoored_encoder/2024-01-05-17:08:16/model_200.pth', 0, './output/stl10/cifar10_backdoored_encoder/2024-01-05-17:08:16/unet_filter_200_trained.pt', 'airplane', 'JPEGcompression','backdoor')# color + loss0
# run_eval(0, 'stl10', 'cifar10', './output/stl10/cifar10_backdoored_encoder/2024-01-05-17:10:47/model_200.pth', 0, './output/stl10/cifar10_backdoored_encoder/2024-01-05-17:10:47/unet_filter_200_trained.pt', 'airplane', 'JPEGcompression','backdoor') # loss0


# run_eval(0, 'stl10', 'cifar10', './output/stl10/cifar10_backdoored_encoder/2024-01-05-17:08:16/model_200.pth', 0, './output/stl10/cifar10_backdoored_encoder/2024-01-05-17:08:16/unet_filter_200_trained.pt', 'airplane', 'poisson_noise', 'backdoor')# color + loss0
# run_eval(0, 'stl10', 'cifar10', './output/stl10/cifar10_backdoored_encoder/2024-01-05-17:10:47/model_200.pth', 0, './output/stl10/cifar10_backdoored_encoder/2024-01-05-17:10:47/unet_filter_200_trained.pt', 'airplane', 'poisson_noise', 'backdoor') # loss0


# run_eval(0, 'stl10', 'cifar10', './output/stl10/cifar10_backdoored_encoder/2024-01-05-17:08:16/model_200.pth', 0, './output/stl10/cifar10_backdoored_encoder/2024-01-05-17:08:16/unet_filter_200_trained.pt', 'airplane', 'None', 'backdoor')# color + loss0
# run_eval(0, 'stl10', 'cifar10', './output/stl10/cifar10_backdoored_encoder/2024-01-05-17:10:47/model_200.pth', 0, './output/stl10/cifar10_backdoored_encoder/2024-01-05-17:10:47/unet_filter_200_trained.pt', 'airplane', 'None', 'backdoor') # loss0
########################### STL-GTSRB

# run_eval(2, 'stl10', 'gtsrb', './output/stl10/gtsrb_backdoored_encoder/2024-01-05-17:09:25/model_200.pth', 12, './output/stl10/gtsrb_backdoored_encoder/2024-01-05-17:09:25/unet_filter_200_trained.pt', 'priority', 'salt_and_pepper_noise', 'backdoor')# color + loss0
# run_eval(2, 'stl10', 'gtsrb', './output/stl10/gtsrb_backdoored_encoder/2024-01-05-17:11:20/model_200.pth', 12, './output/stl10/gtsrb_backdoored_encoder/2024-01-05-17:11:20/unet_filter_200_trained.pt', 'priority', 'salt_and_pepper_noise', 'backdoor') # loss0

# run_eval(5, 'stl10', 'gtsrb', './output/stl10/gtsrb_backdoored_encoder/2024-01-05-17:09:25/model_200.pth', 12, './output/stl10/gtsrb_backdoored_encoder/2024-01-05-17:09:25/unet_filter_200_trained.pt', 'priority', 'poisson_noise', 'backdoor')# color + loss0
# run_eval(0, 'stl10', 'gtsrb', './output/stl10/gtsrb_backdoored_encoder/2024-01-05-17:11:20/model_200.pth', 12, './output/stl10/gtsrb_backdoored_encoder/2024-01-05-17:11:20/unet_filter_200_trained.pt', 'priority', 'poisson_noise', 'backdoor') # loss0

# run_eval(3, 'stl10', 'gtsrb', './output/stl10/gtsrb_backdoored_encoder/2024-01-05-17:09:25/model_200.pth', 12, './output/stl10/gtsrb_backdoored_encoder/2024-01-05-17:09:25/unet_filter_200_trained.pt', 'priority', 'JPEGcompression', 'backdoor')# color + loss0
# run_eval(3, 'stl10', 'gtsrb', './output/stl10/gtsrb_backdoored_encoder/2024-01-05-17:11:20/model_200.pth', 12, './output/stl10/gtsrb_backdoored_encoder/2024-01-05-17:11:20/unet_filter_200_trained.pt', 'priority', 'JPEGcompression', 'backdoor') # loss0

# run_eval(5, 'stl10', 'svhn', './output/stl10/svhn_backdoored_encoder/2024-01-05-17:09:46/model_200.pth', 1, './output/stl10/svhn_backdoored_encoder/2024-01-05-17:09:46/unet_filter_200_trained.pt', 'one', 'backdoor') # color + loss0
# run_eval(5, 'stl10', 'svhn', './output/stl10/svhn_backdoored_encoder/2024-01-05-17:11:23/model_200.pth', 1, './output/stl10/svhn_backdoored_encoder/2024-01-05-17:11:23/unet_filter_200_trained.pt', 'one', 'backdoor') # loss0


# run_eval(1, 'stl10', 'svhn', './output/stl10/svhn_backdoored_encoder/2024-01-04-20:26:54/model_200.pth', 1, './output/stl10/svhn_backdoored_encoder/2024-01-04-20:26:54/unet_filter_200_trained.pt', 'one', 'backdoor') # color + loss0
# run_eval(0, 'stl10', 'svhn', './output/stl10/svhn_backdoored_encoder/2024-01-04-20:28:10/model_200.pth', 1, './output/stl10/svhn_backdoored_encoder/2024-01-04-20:28:10/unet_filter_200_trained.pt', 'one', 'backdoor') # loss0
