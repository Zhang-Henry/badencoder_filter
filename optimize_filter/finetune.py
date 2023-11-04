import sys
sys.path.append("..")

import torch,os,datetime
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from torch.nn import MSELoss,Identity
from torchvision import models
from torchvision.models import resnet50, ResNet50_Weights,vit_l_16,ViT_L_16_Weights,vit_b_16,ViT_B_16_Weights,swin_s,Swin_S_Weights
import numpy as np
from tqdm import tqdm
from utils import *
# from optimize_filter.previous.data_loader import aug
# from simclr_converter.resnet_wider import resnet50x1, resnet50x2, resnet50x4



class Finetuner():
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.best_acc=0
        # self.backbone=resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(self.device).eval()
        # self.backbone=vit_l_16(weights=ViT_L_16_Weights.IMAGENET1K_SWAG_LINEAR_V1).to(self.device).eval()
        # self.backbone=vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1).to(self.device).eval()
        # self.backbone=swin_s(weights=Swin_S_Weights.IMAGENET1K_V1).to(self.device).eval()
        # self.backbone.head=Identity()

        self.backbone = resnet50x1().to(self.device) # simclr
        sd = torch.load('../simclr_converter/resnet50-1x.pth', map_location=torch.device('cuda:0'))
        self.backbone.load_state_dict(sd['state_dict'])

        ############### moco pretrained https://github.com/facebookresearch/moco ##############
        # model = models.__dict__['resnet50']()

        # checkpoint = torch.load('/home/hrzhang/projects/SSL-Backdoor/moco/save/moco_v2_800ep_pretrain.pth.tar')
        # state_dict = checkpoint["state_dict"]

        # for k in list(state_dict.keys()):
        #     # retain only encoder_q up to before the embedding layer
        #     if k.startswith("module.encoder_q") and not k.startswith(
        #         "module.encoder_q.fc"
        #     ):
        #         # remove prefix
        #         state_dict[k[len("module.encoder_q.") :]] = state_dict[k]
        #     # delete renamed or unused k
        #     del state_dict[k]

        # model.load_state_dict(state_dict, strict=False)
        # self.backbone = model.to(self.device).eval()


    def train(self,args,train_loader,val_loader,test_loader):
        self.backbone.fc = nn.Sequential(
            nn.Linear(self.backbone.fc.in_features, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 100),
        )
        self.backbone = self.backbone.to(self.device)

        # for name, param in self.backbone.named_parameters():
        #     if not name.startswith("fc"):
        #         param.requires_grad = False

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.backbone.parameters(), lr=args.lr)
        scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
        bar=tqdm(range(1, args.n_epoch+1))
        for epoch in bar:
            # Training
            self.backbone.train()
            running_loss = 0.0
            for img, label in train_loader:
                img = img.to(self.device)
                label = label.to(self.device)
                optimizer.zero_grad()
                outputs = self.backbone(img)
                loss = criterion(outputs, label)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            scheduler.step()

            # Validation
            self.backbone.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for img, label in val_loader:
                    img = img.to(self.device)
                    label = label.to(self.device)
                    outputs = self.backbone(img)
                    val_loss += criterion(outputs, label).item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += label.size(0)
                    correct += (predicted == label).sum().item()

            bar.set_description(f"Loss: {running_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {100 * correct / total:.2f}%")

            # Testing every 10 epochs
            if epoch % 5 == 0:
                test_acc = self.evaluate(test_loader, criterion)
                if test_acc > self.best_acc:
                    self.best_acc = test_acc
                    torch.save(self.backbone, f'backbone/best_model_acc{self.best_acc:.2f}.pth')  # Save the dictionary

    def evaluate(self, test_loader, criterion):
        # Testing
        self.backbone.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for img, label in test_loader:
                img = img.to(self.device)
                label = label.to(self.device)
                outputs = self.backbone(img)
                test_loss += criterion(outputs, label).item()
                _, predicted = torch.max(outputs.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()
        test_acc=100 * correct / total
        print(f'\nTest Loss: {test_loss/len(test_loader):.4f}, Test Acc: {test_acc:.2f}%\n')
        return test_acc


