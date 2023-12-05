import sys
sys.path.append("..")

import torch,lpips
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from torch.nn import MSELoss
import numpy as np
from pytorch_ssim import SSIM
from tqdm import tqdm
from PIL import Image
from utils import *
from loss import *
from network import U_Net,R2AttU_Net,R2U_Net,AttU_Net
from tiny_network import U_Net_tiny
# from optimize_filter.previous.data_loader import aug
from torchmetrics.image import PeakSignalNoiseRatio


class Solver():
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.filter = torch.nn.Parameter(torch.randn(3, 3, 7, 7, requires_grad=True).cuda())  # 修改滤波器形状并将其放在GPU上
        self.optimizer = torch.optim.Adam([self.filter], lr=args.lr)

        self.scheduler = StepLR(self.optimizer, step_size=args.step_size, gamma=args.gamma)

        self.ssim = SSIM()
        self.psnr = MSELoss()
        # self.loss_fn = lpips.LPIPS(net='alex').to(self.device)
        # self.WD=SinkhornDistance(eps=0.1, max_iter=100)
        self.psnr = PeakSignalNoiseRatio().to(self.device)
        self.color_loss_fn = CombinedColorLoss().to(self.device)


    def train(self,args,train_loader,test_loader):
        print('Start training...')

        bar=tqdm(range(1, args.n_epoch+1))
        recorder=Recorder(args)
        tracker=Loss_Tracker(['loss', 'ssim1', 'psnr1', 'ssim', 'psnr'])
        # tracker_test=Loss_Tracker(['wd', 'ssim', 'psnr', 'lp','color'])


        # 恢复模型和优化器状态
        if args.resume:
            checkpoint = torch.load(args.resume)
            self.net.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            recorder.best = checkpoint['best']
            print(f"\nResuming training from {args.resume}")

        for _ in bar:
            self.train_one_epoch(args,recorder,bar,tracker,train_loader)


    def train_one_epoch(self,args,recorder,bar,tracker,train_loader):
        tracker.reset() # 重置损失记录器

        for img,img_trans in train_loader:
            img = img.to(self.device)
            img_trans = img_trans.to(self.device)

            filter_img = F.conv2d(img, self.filter, padding=7//2)
            # filter_img = torch.clamp(filter_img, 0, 1)

            # sig = nn.Sigmoid()
            # filter_img = sig(filter_img)

            loss_psnr_trans = self.psnr(filter_img, img_trans)
            loss_ssim_trans = self.ssim(filter_img,img_trans)

            # filter后的图片和原图的psnr和ssim，差距要尽可能小,ssim psnr提高
            loss_psnr  = self.psnr(filter_img, img)
            loss_ssim = self.ssim(filter_img, img)


            loss = -(0.025 * loss_psnr + loss_ssim)*10 + (0.025 * loss_psnr_trans + loss_ssim_trans)
            # ssim1和transformed的差距要尽可能大
            # ssim和原图的差距要尽可能小

            self.optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_([self.filter], max_norm=1.0)
            self.optimizer.step()
            losses={'loss':loss.item(),'ssim1':loss_ssim_trans.item(),'psnr1':loss_psnr_trans.item(),'ssim':loss_ssim.item(),'psnr':loss_psnr.item()}
            tracker.update(losses)

        self.scheduler.step()
        # 计算平均损失

        avg_losses = tracker.get_avg_loss()
        avg_loss, ssim_trans, psnr_trans, ssim, psnr = avg_losses.values()

        if ssim >= args.ssim_threshold and ssim_trans >= recorder.best:

            torch.save(self.filter, f'trigger/{args.dataset}/{self.args.timestamp}/ssimtrans{ssim_trans:.4f}_psnr_trans{psnr_trans:.2f}_ssim{ssim:.4f}_psnr{psnr:.2f}.pt')

            recorder.best = ssim_trans
            print('\n--------------------------------------------------')
            print(f"Updated !!! Best SSIM: {ssim_trans}, psnr_trans: {psnr_trans}")
            print('--------------------------------------------------')
            recorder.cost_up_counter = 0
            recorder.cost_down_counter = 0

        bar.set_description(f"Loss: {avg_loss}, lr: {self.optimizer.param_groups[0]['lr']}, SSIM: {ssim:.5f}, psnr:{psnr:.5f}, SSIM_trans: {ssim_trans:.5f}, psnr_trans:{psnr_trans:.5f},  cost:{recorder.cost}")
