import argparse,torch,random,os
import numpy as np
from optimize_filter.previous.data_loader import create_data_loader
from solver import Solver

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

    parser = argparse.ArgumentParser()
    parser.add_argument('--timestamp', type=str)

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_epoch', type=int, default=100)
    parser.add_argument('--step_size', type=int, default=90)
    parser.add_argument('--gamma', type=int, default=0.1)

    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--num_workers', type=int, default=2)

    parser.add_argument('--ssim_threshold', type=float, default=0.95)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--init_cost', type=float, default=1e-5)
    parser.add_argument('--cost_multiplier_up', type=float, default=1.5**1.5)
    parser.add_argument('--cost_multiplier_down', type=float, default=1.5)
    parser.add_argument('--gpu', default='0', type=str, help='the index of gpu used to train the model')

    args = parser.parse_args()
    print(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    print('Loading data...')
    train_loader = create_data_loader(args)
    os.makedirs(f'trigger/moco/{args.timestamp}',exist_ok=True)
    solver=Solver(args,train_loader)
    solver.train(args)

