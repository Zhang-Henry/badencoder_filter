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
            >./log/{encoder_usage_info}/evaluation_{key}_{encoder_usage_info}_{downstream_dataset}_100.log 2>&1 &"


    os.system(cmd)


# run_eval(5, 'cifar10', 'stl10', 'output/cifar10/clean_encoder/model_1000.pth', 9, './trigger/cifar10/unet_filter.pt', 'truck')
# run_eval(5, 'cifar10', 'gtsrb', 'output/cifar10/clean_encoder/model_1000.pth', 12, './trigger/cifar10/unet_filter.pt', 'priority')
# run_eval(5, 'cifar10', 'svhn', 'output/cifar10/clean_encoder/model_1000.pth', 1, './trigger/cifar10/unet_filter.pt', 'one')


# run_eval(0, 'cifar10', 'stl10', 'output/cifar10/stl10_backdoored_encoder/2024-02-22-21:09:53/model_200.pth', 9, 'output/cifar10/stl10_backdoored_encoder/2024-02-22-21:09:53/unet_filter_200_trained.pt', 'truck', 'backdoor') # color + loss0

# run_eval(1, 'cifar10', 'gtsrb', './output/cifar10/gtsrb_backdoored_encoder/2024-01-28-13:17:00/model_100.pth', 12, 'output/cifar10/gtsrb_backdoored_encoder//2024-01-28-13:17:00/unet_filter_100_trained.pt', 'priority', 'backdoor') # color + loss0
# run_eval(0, 'cifar10', 'gtsrb', './output/cifar10/gtsrb_backdoored_encoder/2024-01-03-12:20:49/model_200.pth', 12, 'output/cifar10/gtsrb_backdoored_encoder/2023-12-29-13:20:56/unet_filter_200_trained.pt', 'priority', 'backdoor') # loss0

# run_eval(1, 'cifar10', 'svhn', './output/cifar10/svhn_backdoored_encoder/2024-02-22-21:10:23/model_200.pth', 1, 'output/cifar10/svhn_backdoored_encoder/2024-02-22-21:10:23/unet_filter_200_trained.pt', 'one', 'backdoor') # color + loss0
# run_eval(1, 'cifar10', 'svhn', './output/cifar10/svhn_backdoored_encoder/2024-03-07-20:31:16/model_50.pth', 1, 'output/cifar10/svhn_backdoored_encoder/2024-03-07-20:31:16/unet_filter_50_trained.pt', 'one', 'backdoor') # loss0


# run_eval(1, 'stl10', 'cifar10', ' ./output/stl10/cifar10_backdoored_encoder/2024-02-24-22:52:09/model_200.pth', 0, './output/stl10/cifar10_backdoored_encoder/2024-02-24-22:52:09/unet_filter_100_trained.pt', 'airplane', 'backdoor')# color + loss0

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



## MOCO
# run_eval(6, 'MOCO', 'stl10', 'output/MOCO/stl10_backdoored_encoder/2024-05-14-13:28:11/model_200.pth', 9, 'output/MOCO/stl10_backdoored_encoder/2024-05-14-13:28:11/unet_filter_200_trained.pt', 'truck', 'backdoor')
# run_eval(0, 'MOCO', 'gtsrb', 'output/MOCO/gtsrb_backdoored_encoder/2024-05-13-23:22:46/model_200.pth', 12, 'output/MOCO/gtsrb_backdoored_encoder/2024-05-13-23:22:46/unet_filter_200_trained.pt', 'priority', 'backdoor')
# run_eval(1, 'MOCO', 'svhn', 'output/MOCO/svhn_backdoored_encoder/2024-05-13-23:21:27/model_200.pth', 1, 'output/MOCO/svhn_backdoored_encoder/2024-05-13-23:21:27/unet_filter_200_trained.pt', 'one', 'backdoor')

# run_eval(0, 'MOCO', 'stl10', 'output/cifar10-moco/clean_encoder/moco-model.pth', 9, 'output/MOCO/stl10_backdoored_encoder/2024-05-14-13:28:11/unet_filter_200_trained.pt', 'truck', 'clean')
# run_eval(0, 'MOCO', 'gtsrb', 'output/cifar10-moco/clean_encoder/moco-model.pth', 12, 'output/MOCO/gtsrb_backdoored_encoder/2024-05-13-23:22:46/unet_filter_200_trained.pt', 'priority', 'clean')
# run_eval(1, 'MOCO', 'svhn', 'output/cifar10-moco/clean_encoder/moco-model.pth', 1, 'output/MOCO/svhn_backdoored_encoder/2024-05-13-23:21:27/unet_filter_200_trained.pt', 'one', 'clean')

## CLIP
# run_eval(0, 'CLIP', 'stl10', 'output/CLIP/stl10_backdoored_encoder/2024-05-14-14:06:55/model_40.pth', 9, 'output/CLIP/stl10_backdoored_encoder/2024-05-14-14:06:55/unet_filter_40_trained.pt', 'truck', 'backdoor')
# run_eval(7, 'CLIP', 'gtsrb', 'output/CLIP/gtsrb_backdoored_encoder/2024-05-13-13:21:30/model_40.pth', 12, 'output/CLIP/gtsrb_backdoored_encoder/2024-05-13-13:21:30/unet_filter_40_trained.pt', 'priority', 'backdoor')
# run_eval(4, 'CLIP', 'svhn', 'output/CLIP/svhn_backdoored_encoder/2024-05-14-17:08:40/model_20.pth', 1, 'output/CLIP/svhn_backdoored_encoder/2024-05-14-17:08:40/unet_filter_20_trained.pt', 'one', 'backdoor')


## SWAV
# run_eval(7, 'swav', 'stl10', 'output/swav/stl10_backdoored_encoder/2024-05-15-15:27:39/model_160.pth', 9, 'output/swav/stl10_backdoored_encoder/2024-05-15-15:27:39/unet_filter_160_trained.pt', 'truck', 'backdoor')
# run_eval(4, 'swav', 'gtsrb', 'output/swav/gtsrb_backdoored_encoder/2024-05-14-16:11:44/model_160.pth', 12, 'output/swav/gtsrb_backdoored_encoder/2024-05-14-16:11:44/unet_filter_160_trained.pt', 'priority', 'backdoor')
# run_eval(6, 'swav', 'svhn', 'output/swav/svhn_backdoored_encoder/2024-05-14-16:12:53/model_140.pth', 1, 'output/swav/svhn_backdoored_encoder/2024-05-14-16:12:53/unet_filter_140_trained.pt', 'one', 'backdoor')

# run_eval(1, 'swav', 'svhn', 'log/benchmark_logs/cifar10/version_0/SwaV/checkpoints/epoch=799-step=19200.ckpt', 1, 'output/simsiam/svhn_backdoored_encoder/2024-05-14-15:36:33/unet_filter_100_trained.pt', 'one', 'clean')
# run_eval(7, 'swav', 'stl10', 'log/benchmark_logs/cifar10/version_0/SwaV/checkpoints/epoch=799-step=19200.ckpt', 9, 'output/MOCO/stl10_backdoored_encoder/2024-05-14-13:28:11/unet_filter_140_trained.pt', 'truck', 'clean')
# run_eval(7, 'swav', 'gtsrb', 'log/benchmark_logs/cifar10/version_0/SwaV/checkpoints/epoch=799-step=19200.ckpt', 12, 'output/MOCO/gtsrb_backdoored_encoder/2024-05-13-23:22:46/unet_filter_100_trained.pt', 'priority', 'clean')

## simsiam
# run_eval(6, 'simsiam', 'stl10', 'output/simsiam/stl10_backdoored_encoder/2024-05-14-15:39:35/model_200.pth', 9, 'output/simsiam/stl10_backdoored_encoder/2024-05-14-15:39:35/unet_filter_200_trained.pt', 'truck', 'backdoor')
# run_eval(5, 'simsiam', 'gtsrb', 'output/simsiam/gtsrb_backdoored_encoder/2024-05-14-15:38:13/model_80.pth', 12, 'output/simsiam/gtsrb_backdoored_encoder/2024-05-14-15:38:13/unet_filter_80_trained.pt', 'priority', 'backdoor')
# run_eval(0, 'simsiam', 'svhn', 'output/simsiam/svhn_backdoored_encoder/2024-05-14-15:36:33/model_200.pth', 1, 'output/simsiam/svhn_backdoored_encoder/2024-05-14-15:36:33/unet_filter_200_trained.pt', 'one', 'backdoor')

# run_eval(4, 'simsiam', 'svhn', 'log/benchmark_logs/cifar10/version_0/SimSiam/checkpoints/epoch=799-step=19200.ckpt', 1, 'output/simsiam/svhn_backdoored_encoder/2024-05-14-15:36:33/unet_filter_100_trained.pt', 'one', 'clean')
# run_eval(5, 'simsiam', 'gtsrb', 'log/benchmark_logs/cifar10/version_0/SimSiam/checkpoints/epoch=799-step=19200.ckpt', 12, 'output/simsiam/gtsrb_backdoored_encoder/2024-05-14-15:38:13/unet_filter_80_trained.pt', 'priority', 'clean')
# run_eval(6, 'simsiam', 'stl10', 'log/benchmark_logs/cifar10/version_0/SimSiam/checkpoints/epoch=799-step=19200.ckpt', 9, 'output/simsiam/stl10_backdoored_encoder/2024-05-14-15:39:35/unet_filter_100_trained.pt', 'truck', 'clean')

## byol
# run_eval(2, 'byol', 'stl10', 'output/byol/stl10_backdoored_encoder/2024-05-15-16:38:57/model_200.pth', 9, 'output/byol/stl10_backdoored_encoder/2024-05-15-16:38:57/unet_filter_200_trained.pt', 'truck', 'backdoor')
# run_eval(2, 'byol', 'gtsrb', 'output/byol/svhn_backdoored_encoder/2024-05-15-16:38:06/model_200.pth', 12, 'output/byol/svhn_backdoored_encoder/2024-05-15-16:38:06/unet_filter_200_trained.pt', 'priority', 'backdoor')
# run_eval(2, 'byol', 'svhn', 'output/byol/svhn_backdoored_encoder/2024-05-15-16:38:06/model_100.pth', 1, 'output/byol/svhn_backdoored_encoder/2024-05-15-16:38:06/unet_filter_100_trained.pt', 'one', 'backdoor')


## NNCLR
# run_eval(2, 'NNCLR', 'stl10', 'output/NNCLR/stl10_backdoored_encoder/2024-05-15-23:22:13/model_100.pth', 9, 'output/NNCLR/stl10_backdoored_encoder/2024-05-15-23:22:13/unet_filter_100_trained.pt', 'truck', 'backdoor')
run_eval(2, 'NNCLR', 'gtsrb', 'output/NNCLR/gtsrb_backdoored_encoder/2024-05-15-23:22:20/model_100.pth', 12, 'output/NNCLR/gtsrb_backdoored_encoder/2024-05-15-23:22:20/unet_filter_100_trained.pt', 'priority', 'backdoor')
# run_eval(2, 'NNCLR', 'svhn', 'output/NNCLR/svhn_backdoored_encoder/2024-05-15-23:21:12/model_200.pth', 1, 'output/NNCLR/svhn_backdoored_encoder/2024-05-15-23:21:12/unet_filter_200_trained.pt', 'one', 'backdoor')


## DINO
# run_eval(5, 'DINO', 'stl10', 'output/DINO/stl10_backdoored_encoder/2024-05-16-08:41:02/model_140.pth', 9, 'output/DINO/stl10_backdoored_encoder/2024-05-16-08:41:02/unet_filter_140_trained.pt', 'truck', 'backdoor')
# run_eval(5, 'DINO', 'gtsrb', 'output/DINO/gtsrb_backdoored_encoder/2024-05-16-08:41:06/model_120.pth', 12, 'output/DINO/gtsrb_backdoored_encoder/2024-05-16-08:41:06/unet_filter_120_trained.pt', 'priority', 'backdoor')
# run_eval(5, 'DINO', 'svhn', 'output/DINO/svhn_backdoored_encoder/2024-05-16-08:40:30/model_60.pth', 1, 'output/DINO/svhn_backdoored_encoder/2024-05-16-08:40:30/unet_filter_60_trained.pt', 'one', 'backdoor')