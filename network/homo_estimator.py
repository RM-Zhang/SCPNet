import torch
import torch.nn as nn
import torch.nn.functional as F
import torchgeometry as tgm
import sys
sys.path.append("./")
from network.utils_net import *
from network.corr import CorrBlock


class Encoder_32(nn.Module):
    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0):
        super(Encoder_32, self).__init__()
        self.norm_fn = norm_fn

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)
        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)
        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(96, stride=1)

        # output convolution
        self.convout2 = nn.Conv2d(96, output_dim, kernel_size=1)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        
        x = F.max_pool2d(x, 2, stride=2)
        x = self.layer1(x)

        x = F.max_pool2d(x, 2, stride=2)
        x = self.layer2(x)

        x_32 = self.convout2(x)

        if self.training and self.dropout is not None:
            x = self.dropout(x)

        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)

        return x_32


class Decoder_32(nn.Module):
    def __init__(self, input_dim=256):
        super(Decoder_32, self).__init__()
        outputdim = input_dim
        self.layer1 = nn.Sequential(nn.Conv2d(164, outputdim, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=outputdim//8, num_channels=outputdim), nn.ReLU(), nn.MaxPool2d(kernel_size = 2, stride=2))

        input_dim = outputdim
        outputdim = input_dim
        self.layer2 = nn.Sequential(nn.Conv2d(input_dim, outputdim, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=(outputdim) // 8, num_channels=outputdim), nn.ReLU(), nn.MaxPool2d(kernel_size = 2, stride=2))
        input_dim = outputdim
        outputdim = input_dim
        self.layer3 = nn.Sequential(nn.Conv2d(input_dim, outputdim, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=(outputdim) // 8, num_channels=outputdim), nn.ReLU(),
                                    nn.MaxPool2d(kernel_size = 2, stride=2))
        input_dim = outputdim
        outputdim = input_dim
        self.layer4 = nn.Sequential(nn.Conv2d(input_dim, outputdim, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=(outputdim) // 8, num_channels=outputdim), nn.ReLU(),
                                    nn.MaxPool2d(kernel_size = 2, stride=2))

        input_dim = outputdim
        outputdim_final = outputdim

        ### global motion
        self.layer10 = nn.Sequential(nn.Conv2d(input_dim, outputdim_final, 3,  padding=1, stride=1), nn.GroupNorm(num_groups=(outputdim_final) // 8, num_channels=outputdim_final),
                                    nn.ReLU(), nn.Conv2d(outputdim_final, 2, 1))
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer10(x)
        return x
    
    
class Homo_estimator_32(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.device = torch.device('cuda:' + str(args.gpuid[0]))
        self.args = args
        self.fnet = Encoder_32(output_dim=256, norm_fn='instance')
        self.update_block = Decoder_32(input_dim=128)

    def initialize_flow_4(self, img):
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//4, W//4).to(img.device)
        coords1 = coords_grid(N, H//4, W//4).to(img.device)
        return coords0, coords1

    def forward(self, image1, image2):
        image1 = image1.contiguous()
        image2 = image2.contiguous()
        fmap1 = self.fnet(image1).float() #[B, 256, 32, 32]
        fmap2 = self.fnet(image2).float()
        
        corr_fn = CorrBlock(fmap1, fmap2, num_levels=2, radius=4, sz=32)
        coords0, coords1 = self.initialize_flow_4(image1)
        sz = fmap1.shape
        self.sz = sz

        corr = corr_fn(coords1)  # [B, 162, 32, 32]
        flow = coords1 - coords0 # [B, 2, 32, 32]
        delta_four_point = self.update_block(torch.cat((corr, flow), dim=1)) # [B, 2, 2, 2]

        four_point_reshape = delta_four_point.permute(0,2,3,1).reshape(-1,4,2) # [top_left, top_right, bottom_left, bottom_right], [-1, 4, 2]
        return four_point_reshape
