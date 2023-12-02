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
        self.mse = MSELoss()
        self.loss_fn = lpips.LPIPS(net='alex').to(self.device)
        self.WD=SinkhornDistance(eps=0.1, max_iter=100)
        self.psnr = PeakSignalNoiseRatio().to(self.device)
        self.color_loss_fn = CombinedColorLoss().to(self.device)
        # self.backbone = load_backbone()
        # self.backbone = self.backbone.to(self.device).eval()
        # print(self.backbone)
        # print(self.net)


    def train(self,args,train_loader,test_loader):
        print('Start training...')

        bar=tqdm(range(1, args.n_epoch+1))
        recorder=Recorder(args)
        tracker=Loss_Tracker(['loss', 'ssim1', 'mse1', 'ssim', 'mse'])
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
            # self.test_one_epoch(args,test_loader,tracker_test)

    # def test_one_epoch(self,args,test_loader,tracker_test):
    #     for img,img_trans in test_loader:
    #         img = img.to(self.device)
    #         img_trans = img_trans.to(self.device)

    #         # 将滤镜作用在Aug的图像上
    #         filter_img = self.net(img_trans)
    #         filter_img = self.net(img)


    #         # sig=torch.nn.Sigmoid()
    #         # filter_img = sig(filter_img)
    #         # color_loss = self.color_loss_fn(filter_img, img_trans)
    #         img_trans_feature = self.backbone(img_trans)
    #         filter_img_feature = self.backbone(filter_img)

    #         img_trans_feature = F.normalize(img_trans_feature, dim=-1)
    #         filter_img_feature = F.normalize(filter_img_feature, dim=-1)
    #         wd,_,_=self.WD(filter_img_feature,img_trans_feature) # wd越小越相似，拉远backdoor img和transformed backdoor img的距离

    #         loss_psnr = self.psnr(filter_img, img)
    #         loss_ssim = self.ssim(filter_img, img)

    #         d_list = self.loss_fn(filter_img,img)
    #         lp_loss=d_list.squeeze()

    #         losses={'wd':wd.item(),'ssim':loss_ssim.item(),'psnr':loss_psnr.item(),'lp':lp_loss.mean().item(),'color':color_loss.item()}
    #         tracker_test.update(losses)

    #     avg_losses = tracker_test.get_avg_loss()
    #     wd, ssim, psnr, lp, color = avg_losses.values()
    #     print(f"\nTEST: WD: {wd:.5f}, SSIM: {ssim:.5f}, pnsr:{psnr:.5f}, lp:{lp:.5f}, color:{color:.5f}")


    def train_one_epoch(self,args,recorder,bar,tracker,train_loader):
        tracker.reset() # 重置损失记录器

        for img,img_trans in train_loader:
            img = img.to(self.device)
            img_trans = img_trans.to(self.device)

            # 将滤镜作用在Aug的图像上
            # filter_img = self.net(img_trans)
            # filter_img = self.net(img)

            filter_img = F.conv2d(img, self.filter, padding=7//2)

            # img_trans_feature = self.backbone(img_trans)
            # filter_img_feature = self.backbone(filter_img)

            # img_trans_feature = F.normalize(img_trans_feature, dim=-1)
            # filter_img_feature = F.normalize(filter_img_feature, dim=-1)
            # wd,_,_=self.WD(filter_img_feature,img_trans_feature) # wd越小越相似，拉远backdoor img和transformed backdoor img的距离
            # wd = compute_style_loss(filter_img_feature,img_trans_feature)

            # filter后的图片和原图的mse和ssim，差距要尽可能小

            # loss_psnr_trans = self.psnr(filter_img, img_trans)
            loss_ssim_trans = self.ssim(filter_img,img_trans)
            loss_mse_trans =  self.mse(filter_img,img_trans)

            # loss_psnr  = self.psnr(filter_img, img)
            loss_ssim = self.ssim(filter_img, img)
            loss_mse =  self.mse(filter_img,img)


            # d_list = self.loss_fn(filter_img,img)
            # lp_loss=d_list.squeeze()


            # loss = -0.01*loss_mse_trans + loss_ssim_trans - (-0.01*loss_mse + loss_ssim)
            loss = 10*(0.001 * loss_mse + 1 - loss_ssim) - (0.001 * loss_mse_trans + 1 - loss_ssim_trans)
            # ssim1和transformed的差距要尽可能大
            # ssim和原图的差距要尽可能小

            self.optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_([self.filter], max_norm=1.0)
            self.optimizer.step()
            losses={'loss':loss.item(),'ssim1':loss_ssim_trans.item(),'mse1':loss_mse_trans.item(),'ssim':loss_ssim.item(),'mse':loss_mse.item()}
            tracker.update(losses)

        self.scheduler.step()
        # 计算平均损失

        avg_losses = tracker.get_avg_loss()
        avg_loss, ssim_trans, mse_trans, ssim, mse = avg_losses.values()

        # if args.ablation:
        #     if ssim >= args.ssim_threshold and ssim < args.ssim_threshold+0.01:
        #         state = {
        #             'model_state_dict': self.net.state_dict(),
        #             'optimizer_state_dict': self.optimizer.state_dict(),
        #             'best': recorder.best
        #         }
        #         torch.save(state, f'trigger/{args.dataset}/{self.args.timestamp}/ablation_ssim{ssim:.4f}_psnr{psnr:.2f}_lp{lp:.4f}_wd{wd:.3f}_color{color:.3f}.pt')

        #         recorder.best = color
        #         print('\n--------------------------------------------------')
        #         print(f"Updated !!! Best sim:{sim}, far:{far}, SSIM: {ssim}, psnr: {psnr}, lp: {lp}, WD: {wd}, color: {color}")
        #         print('--------------------------------------------------')
        #         recorder.cost_up_counter = 0
        #         recorder.cost_down_counter = 0

        #     if ssim >= args.ssim_threshold and psnr >= args.psnr_threshold and lp <= args.lp_threshold:
        #         recorder.cost_up_counter += 1
        #         recorder.cost_down_counter = 0
        #     else:
        #         recorder.cost_up_counter = 0
        #         recorder.cost_down_counter += 1

        #     if recorder.cost_up_counter >= args.patience:
        #         recorder.cost_up_counter = 0
        #         print('\n--------------------------------------------------')
        #         print("Up cost from {} to {}".format(recorder.cost, recorder.cost * recorder.cost_multiplier_up))
        #         print('--------------------------------------------------')

        #         recorder.cost *= recorder.cost_multiplier_up
        #         recorder.cost_up_flag = True

        #     elif recorder.cost_down_counter >= args.patience:
        #         recorder.cost_down_counter = 0
        #         print('\n--------------------------------------------------')
        #         print("Down cost from {} to {}".format(recorder.cost, recorder.cost / recorder.cost_multiplier_down))
        #         print('--------------------------------------------------')
        #         recorder.cost /= recorder.cost_multiplier_down
        #         recorder.cost_down_flag = True
        # else: # 正常情况
        if ssim >= args.ssim_threshold and ssim_trans >= recorder.best:
            # state = {
            #     'model_state_dict': self.filter.state_dict(),
            #     'optimizer_state_dict': self.optimizer.state_dict(),
            #     'best': recorder.best
            # }
            torch.save(self.filter, f'trigger/{args.dataset}/{self.args.timestamp}/ssimtrans{ssim_trans:.4f}_mse_trans{mse_trans:.2f}_ssim{ssim:.4f}_mse{mse:.2f}.pt')

            recorder.best = ssim_trans
            print('\n--------------------------------------------------')
            print(f"Updated !!! Best SSIM: {ssim_trans}, mse_trans: {mse_trans}")
            print('--------------------------------------------------')
            recorder.cost_up_counter = 0
            recorder.cost_down_counter = 0


            # if ssim >= args.ssim_threshold and psnr >= args.psnr_threshold and lp <= args.lp_threshold:
            #     recorder.cost_up_counter += 1
            #     recorder.cost_down_counter = 0
            # else:
            #     recorder.cost_up_counter = 0
            #     recorder.cost_down_counter += 1

            # if recorder.cost_up_counter >= args.patience:
            #     recorder.cost_up_counter = 0
            #     print('\n--------------------------------------------------')
            #     print("Up cost from {} to {}".format(recorder.cost, recorder.cost * recorder.cost_multiplier_up))
            #     print('--------------------------------------------------')

            #     recorder.cost *= recorder.cost_multiplier_up
            #     recorder.cost_up_flag = True

            # elif recorder.cost_down_counter >= args.patience:
            #     recorder.cost_down_counter = 0
            #     print('\n--------------------------------------------------')
            #     print("Down cost from {} to {}".format(recorder.cost, recorder.cost / recorder.cost_multiplier_down))
            #     print('--------------------------------------------------')
            #     recorder.cost /= recorder.cost_multiplier_down
            #     recorder.cost_down_flag = True


        # if args.use_feature:
        bar.set_description(f"Loss: {avg_loss}, lr: {self.optimizer.param_groups[0]['lr']}, SSIM: {ssim:.5f}, mse:{mse:.5f}, SSIM_trans: {ssim_trans:.5f}, mse_trans:{mse_trans:.5f},  cost:{recorder.cost}")

        # bar.set_description(f"Loss: {avg_loss}, lr: {self.optimizer.param_groups[0]['lr']}, SIM: {sim:.5f}, far:{far:.5f}, WD: {wd:.8f}, SSIM: {ssim:.5f}, pnsr:{psnr:.5f}, lp:{lp:.5f},mse:{mse:.5f},  cost:{recorder.cost}")
        # else:
        #     bar.set_description(f"Loss: {avg_loss}, lr: {self.optimizer.param_groups[0]['lr']}, SIM: {sim}, far:{far}, WD: {wd},  SSIM: {ssim}, cost:{recorder.cost}, lp:{lp.mean()}")

