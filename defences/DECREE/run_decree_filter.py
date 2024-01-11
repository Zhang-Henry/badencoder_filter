import os
from os import listdir
from os.path import isfile, join

def run(gpu, model_flag, mask_init, enc_path, arch, result_file,
    encoder_usage_info, lr=0.5, batch_size=128, id='_', seed=80):
    det_log_dir = 'detect_log'
    if not os.path.exists(det_log_dir):
        os.makedirs(det_log_dir)
    print(f'log dir: {det_log_dir}')
    cmd = f'nohup python3 -u main.py --gpu {gpu} \
            --model_flag {model_flag} \
            --batch_size {batch_size} \
            --lr {lr} \
            --seed {seed} \
            --encoder_path {enc_path} \
            --mask_init {mask_init} \
            --id {id} \
            --encoder_usage_info {encoder_usage_info} \
            --arch {arch} \
            --result_file {result_file} \
            > {det_log_dir}/cf10_{seed}_{model_flag}_lr{lr}_b{batch_size}_{mask_init}{id}.log 2>&1 &'
    print('cmd: ', cmd)
    os.system(cmd)

gpu = 5
dir_list = [
        '/home/hrzhang/projects/badencoder_filter/output/cifar10/stl10_backdoored_encoder/2023-12-25-20:38:31',
        ]

for dir in dir_list:
    dir_file_list = listdir(dir)
    encoder_list = [f for f in dir_file_list if (isfile(join(dir, f)) and (f.find('.pth') != -1))]
    for encoder in encoder_list:
        enc_path = f'{dir}/{encoder}'
        id = f'_{dir.split("/")[1]}_{dir.split("/")[2].split("_")[0]}_{encoder}'
        flag = 'clean' if 'clean' in dir else 'backdoor'

        ### run CLIP_text
        result_file = 'resultfinal_filter_txt.txt'
        arch= 'resnet18'
        run(gpu, flag, 'rand', lr=0.5, batch_size = 256, encoder_usage_info='cifar10',
            enc_path=enc_path, arch=arch, result_file=result_file, id=id)
