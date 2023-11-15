import argparse,torch,random,os
import numpy as np
from solver import Solver
from finetune import Finetuner
from datetime import datetime
from data_loader import cifar10_dataloader,imagenet_dataloader

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    #torch.use_deterministic_algorithms(True)  # 有检查操作，看下文区别



if __name__ == '__main__':
    # seed_torch()
    now = datetime.now()
    formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")

    print("Start Time:", formatted_time)

    parser = argparse.ArgumentParser()
    parser.add_argument('--timestamp', type=str)
    parser.add_argument('--ssim_threshold', type=float, default=0.95)
    parser.add_argument('--psnr_threshold', type=float, default=30.0)
    parser.add_argument('--lp_threshold', type=float, default=0.003)

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--n_epoch', type=int, default=100)
    parser.add_argument('--step_size', type=int, default=90)
    parser.add_argument('--gamma', type=int, default=0.1)

    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--num_workers', type=int, default=2)

    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--init_cost', type=float, default=1e-5)
    parser.add_argument('--init_cost2', type=float, default=1)
    parser.add_argument('--cost_multiplier_up', type=float, default=1.5**1.5)
    parser.add_argument('--cost_multiplier_down', type=float, default=1.5)
    parser.add_argument('--gpu', default=0, type=int, help='the index of gpu used to train the model')

    parser.add_argument('--ablation', type=bool, default=False)
    parser.add_argument('--num', type=float, default=0.05)
    parser.add_argument('--use_feature', action='store_true',help='use feature or not')
    parser.add_argument('--resume', type=str)
    parser.add_argument('--mode', type=str,choices=['train_filter','finetune_backbone'],default='train_filter')
    parser.add_argument('--max_cost', type=float, default=1e-2)
    parser.add_argument('--min_cost', type=float, default=1e-3)

    args = parser.parse_args()
    print(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    print('Loading data...')
    if args.mode == 'train_filter':
        train_loader = cifar10_dataloader(args)
        os.makedirs(f'trigger/{args.timestamp}',exist_ok=True)
        solver=Solver(args)
        # solver=Solver_ab(args,train_loader)
        solver.train(args,train_loader)

    elif args.mode == 'finetune_backbone':
        # train_loader,val_loader,test_loader = create_finetune_dataloader(args)

        # finetuner=Finetuner(args)
        # finetuner.train(args,train_loader,val_loader,test_loader)
        pass