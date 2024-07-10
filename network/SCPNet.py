import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchgeometry as tgm
import sys
sys.path.append("./")
from network.consistent_projection import *
from network.homo_estimator import *


class SCPNet(nn.Module):
    def __init__(self, args):
        super(SCPNet, self).__init__()
        self.device = torch.device('cuda:' + str(args.gpuid[0]))
        self.args = args
        self.img_projector = CFMP()
        self.homo_predictor = Homo_estimator_32(args=self.args)

    def network_forward(self, img_projector, homo_predictor, img1, img2):
        fmap1 = img_projector(img1)
        fmap2 = img_projector(img2)
        pred_4p = homo_predictor(fmap1, fmap2)
        return fmap1, fmap2, pred_4p
        
    def forward(self, batch_in, mode = 'train'):
        # forward
        pair12_f_w, pair12_f_nw, pair12_pred_4p = self.network_forward(self.img_projector, self.homo_predictor, batch_in['pair12_patch_w'], batch_in['pair12_patch_nw'])
        pair11_f_w, pair11_f_nw, pair11_pred_4p = self.network_forward(self.img_projector, self.homo_predictor, batch_in['pair11_patch_w'], batch_in['pair11_patch_nw'])
        pair22_f_w, pair22_f_nw, pair22_pred_4p = self.network_forward(self.img_projector, self.homo_predictor, batch_in['pair22_patch_w'], batch_in['pair22_patch_nw'])

        # mace
        mace12 = self.calculate_ace(pair12_pred_4p, batch_in['gt12'])
        mace11 = self.calculate_ace(pair11_pred_4p, batch_in['gt11'])
        mace22 = self.calculate_ace(pair22_pred_4p, batch_in['gt22'])

        if mode=='test':
            return {'mace12':mace12, 'mace11':mace11, 'mace22':mace22,
                    }
        else:  
            # intra-model supervised loss
            if self.args.supervised == 'True':
                loss_pair11 = self.sequence_loss(pair11_pred_4p, batch_in['gt11'])
                loss_pair22 = self.sequence_loss(pair22_pred_4p, batch_in['gt22'])
            else:
                loss_pair11 = torch.tensor([0.0]).to(self.device)
                loss_pair22 = torch.tensor([0.0]).to(self.device)
                
            # inter-model unsupervised loss
            if self.args.unsupervised == 'True':
                loss_pair12 = self.unsupervised_loss(pair12_f_w, pair12_f_nw, batch_in['pair12_patch_w'], batch_in['pair12_patch_nw'], pair12_pred_4p)
            else:
                loss_pair12 = torch.tensor([0.0]).to(self.device)
            
            return {'mace12':mace12, 'mace11':mace11, 'mace22':mace22,
                    'loss_pair12':loss_pair12, 'loss_pair11':loss_pair11, 'loss_pair22':loss_pair22
                    }
    
    def unsupervised_loss(self, f_w, f_nw, img_w, img_nw, pred_4p):
        [B,_,H,W] = f_w.shape
        org_pts = torch.tensor([[[0, 0], [W-1, 0], [0, H-1], [W-1, H-1]]], dtype=torch.float32, device=self.device).expand(B, -1, -1)
        dst_pts = org_pts + pred_4p
        H_w_pred = tgm.get_perspective_transform(org_pts, dst_pts)
        H_nw_pred = torch.inverse(H_w_pred)
        f_w_pred = tgm.warp_perspective(f_w.clone().detach(), H_w_pred, (H, W))    # align with f_nw
        f_nw_pred = tgm.warp_perspective(f_nw.clone().detach(), H_nw_pred, (H, W)) # align with f_w
        loss = self.cal_unsuper_l1_loss(f_w.clone().detach(), f_w_pred, f_nw.clone().detach()) +\
                self.cal_unsuper_l1_loss(f_nw.clone().detach(), f_nw_pred, f_w.clone().detach())
        return loss
    
    def cal_unsuper_l1_loss(self, f_w, f_w_pred, f_nw):
        loss = torch.sum(torch.abs(f_w_pred - f_nw))/torch.sum(1e-10 + torch.abs(f_w - f_nw))
        return loss
    
    def sequence_loss(self, pred_4p, gt_4p):
        loss = F.l1_loss(pred_4p, gt_4p)
        return loss

    def calculate_ace(self, pred_4p, gt_4p):
        ace = ((pred_4p - gt_4p)**2).sum(dim=-1).sqrt().mean(dim=-1)
        return ace
