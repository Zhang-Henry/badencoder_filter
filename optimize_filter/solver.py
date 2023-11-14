import sys
sys.path.append("..")
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch,lpips
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from torch.nn import MSELoss
import numpy as np
from pytorch_ssim import SSIM
from tqdm import tqdm
from PIL import Image
from utils import *
from network import U_Net,R2AttU_Net,R2U_Net,AttU_Net
# from optimize_filter.previous.data_loader import aug
from torchmetrics.image import PeakSignalNoiseRatio


class Solver():
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = AttU_Net(img_ch=3,output_ch=3).to(self.device)
        self.optimizer = torch.optim.Adam(list(self.net.parameters()), lr=args.lr, betas=(args.beta1, args.beta2), eps=args.epsilon)
        self.scheduler = StepLR(self.optimizer, step_size=args.step_size, gamma=args.gamma)

        self.mse = MSELoss()
        self.ssim = SSIM()
        # self.loss_cmd = CMD()
        self.loss_fn = lpips.LPIPS(net='alex').to(self.device)
        self.loss_mmd = MMD_loss()
        self.WD=SinkhornDistance(eps=0.1, max_iter=100)
        self.psnr = PeakSignalNoiseRatio().to(self.device)
        self.backbone = load_backbone()
        self.backbone = self.backbone.to(self.device).eval()
        print(self.backbone)
        print(self.net)


    def train(self,args,train_loader):
        print('Start training...')

        bar=tqdm(range(1, args.n_epoch+1))
        recorder=Recorder(args)
        tracker=Loss_Tracker()

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

            # 将滤镜作用在Aug的图像上
            # filter_img = self.net(img_trans)
            filter_img = self.net(img)

            # filter_img = torch.clamp(filter_img, min=0, max=1)

            if args.use_feature:
                with torch.no_grad():
                    img_trans_feature = self.backbone(img_trans)
                    filter_img_feature = self.backbone(filter_img)

                    img_trans_feature = F.normalize(img_trans_feature, dim=1)
                    filter_img_feature = F.normalize(filter_img_feature, dim=1)
                    wd,_,_=self.WD(filter_img_feature,img_trans_feature) # wd越小越相似，拉远backdoor img和transformed backdoor img的距离

                    # wd = self.compute_style_loss(filter_img_feature,img_trans_feature)
            else:
                wd,_,_=self.WD(filter_img.view(filter_img.shape[0],-1),img_trans.view(img_trans.shape[0],-1)) # wd越小越相似

            # cmd=loss_cmd(filter_img.view(aug_filter_img.shape[0],-1),aug_filter_img.view(aug_filter_img.shape[0],-1),5)

            # mmd=loss_mmd(filter_img.view(aug_filter_img.shape[0],-1),aug_filter_img.view(aug_filter_img.shape[0],-1))


            # filter后的图片和原图的mse和ssim，差距要尽可能小
            loss_mse = self.mse(filter_img, img)

            # mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1).cuda()
            # std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1).cuda()

            # filter_img = filter_img * std + mean # denormalize
            # img = img * std + mean

            loss_psnr = self.psnr(filter_img, img)
            loss_ssim = self.ssim(filter_img, img)


            d_list = self.loss_fn(filter_img,img)
            lp_loss=d_list.squeeze()

            ############################ wd ############################
            if args.use_feature:
                loss_sim = 1 - loss_ssim + 10 * lp_loss.mean() - 0.025 * loss_psnr
                loss_far = - recorder.cost * wd
                loss = loss_sim + loss_far
            else:
                loss_sim = 1 - loss_ssim + 10 * lp_loss.mean() - 0.025 * loss_psnr
                loss_far = - recorder.cost * wd
                loss = loss_sim

            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()
            tracker.update(loss.item(),wd.item(),loss_ssim.item(),loss_psnr.item(),lp_loss.mean().item(),loss_mse.item(),loss_sim.item(),loss_far.item())

        self.scheduler.step()
        # 计算平均损失

        avg_loss,wd,ssim,psnr,lp,mse,sim,far = tracker.get_avg_loss()


        # torch.save(self.net, f'trigger/moco/{self.args.timestamp}/ssim{ssim:.4f}_wd{wd:.1f}.pt')
        if ssim >= args.ssim_threshold and psnr >= args.psnr_threshold and lp <= args.lp_threshold and wd >= recorder.best:
            state = {
                'model_state_dict': self.net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best': recorder.best
            }
            torch.save(state, f'trigger/{self.args.timestamp}/ssim{ssim:.4f}_psnr{psnr:.2f}_lp{lp:.4f}_wd{wd:.3f}.pt')

            recorder.best = wd
            print('\n--------------------------------------------------')
            print(f"Updated !!! Best sim:{sim}, far:{far}, SSIM: {ssim}, psnr: {psnr}, lp: {lp}, Best WD: {wd}")
            print('--------------------------------------------------')
            recorder.cost_up_counter = 0
            recorder.cost_down_counter = 0


        if ssim >= args.ssim_threshold and psnr >= args.psnr_threshold and lp <= args.lp_threshold:
            recorder.cost_up_counter += 1
            recorder.cost_down_counter = 0
        else:
            recorder.cost_up_counter = 0
            recorder.cost_down_counter += 1

        if recorder.cost_up_counter >= args.patience:
            recorder.cost_up_counter = 0
            print('\n--------------------------------------------------')
            print("Up cost from {} to {}".format(recorder.cost, recorder.cost * recorder.cost_multiplier_up))
            print('--------------------------------------------------')

            recorder.cost *= recorder.cost_multiplier_up
            recorder.cost_up_flag = True

        elif recorder.cost_down_counter >= args.patience:
            recorder.cost_down_counter = 0
            print('\n--------------------------------------------------')
            print("Down cost from {} to {}".format(recorder.cost, recorder.cost / recorder.cost_multiplier_down))
            print('--------------------------------------------------')
            recorder.cost /= recorder.cost_multiplier_down
            recorder.cost_down_flag = True


        # if args.use_feature:
        bar.set_description(f"Loss: {avg_loss}, lr: {self.optimizer.param_groups[0]['lr']}, SIM: {sim:.5f}, far:{far}, WD: {wd}, SSIM: {ssim:.5f}, pnsr:{psnr:.5f}, lp:{lp:.5f},mse:{mse:.5f},  cost:{recorder.cost}")

        # bar.set_description(f"Loss: {avg_loss}, lr: {self.optimizer.param_groups[0]['lr']}, SIM: {sim:.5f}, far:{far:.5f}, WD: {wd:.8f}, SSIM: {ssim:.5f}, pnsr:{psnr:.5f}, lp:{lp:.5f},mse:{mse:.5f},  cost:{recorder.cost}")
        # else:
        #     bar.set_description(f"Loss: {avg_loss}, lr: {self.optimizer.param_groups[0]['lr']}, SIM: {sim}, far:{far}, WD: {wd},  SSIM: {ssim}, cost:{recorder.cost}, lp:{lp.mean()}")


    def compute_style_loss(self, generated_features, style_features):
        style_loss = 0.0
        for gen_feat, style_feat in zip(generated_features, style_features):
            G_gen = gram_matrix(gen_feat)
            G_style = gram_matrix(style_feat)

            # G_gen = gen_feat.view(gen_feat.shape[0],-1)
            # G_style = style_feat.view(style_feat.shape[0],-1)
            wd,_,_=self.WD(G_gen,G_style)
            # wd,_,_=self.WD(gen_feat,style_feat)
            style_loss += torch.nn.functional.mse_loss(G_gen, G_style)
            style_loss += wd
        return style_loss/5


    def compute_euclidean_loss(self, generated_features, style_features):
        loss = 0.0
        for gen_feat, style_feat in zip(generated_features, style_features):
            gen_feat=F.normalize(gen_feat,dim=1)
            style_feat=F.normalize(style_feat,dim=1)
            dis = self.euclidean_distance(gen_feat,style_feat)
            loss += dis

        return loss/3

    def euclidean_distance(self, img1, img2):
        return torch.sqrt(torch.sum((img1 - img2) ** 2))