import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from msd_pytorch.msd_model import (MSDModel)


# This code is copied and adapted from:
# https://github.com/milesial/Pytorch-UNet

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2), diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self, c_in, c_out, n_filters=64):
        super(UNet, self).__init__()
        self.inc = inconv(c_in, n_filters)
        self.down1 = down(n_filters, n_filters *2)
        self.down2 = down(n_filters *2, n_filters *4)
        self.down3 = down(n_filters *4, n_filters *8)
        self.down4 = down(n_filters *8, n_filters *8)
        self.up1 = up(n_filters *16, n_filters *4)
        self.up2 = up(n_filters *8, n_filters *2)
        self.up3 = up(n_filters *4, n_filters)
        self.up4 = up(n_filters *2, n_filters)
        self.outc = outconv(n_filters, c_out)

    def forward(self, x):
        H, W = x.shape[2:]
        Hp, Wp = ((-H % 16), (-W % 16))
        padding = (Wp // 2, Wp - Wp // 2, Hp // 2, Hp - Hp // 2)
        reflect = nn.ReflectionPad2d(padding)
        x = reflect(x)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)

        H2 = H + padding[2] + padding[3]
        W2 = W + padding[0] + padding[1]
        return x[:, :, padding[2]:H2-padding[3], padding[0]:W2-padding[1]]

    def clear_buffers(self):
        pass

class UNetRegressionModel(MSDModel):
    
    def __init__(self, c_in, c_out, *, width=64, reflect=False, conv3d=False, **kwargs):
        # Initialize msd network.
        super().__init__(c_in, c_out, 1, 1)

        # loss_functions = {'L1': nn.L1Loss(),'L2': nn.MSELoss()}
    
        # self.loss_function = loss
        # self.criterion = loss_functions[loss]
        # assert(self.criterion is not None)

        # Make Unet
        self.msd = UNet(c_in, c_out, width)

        # Initialize network
        self.net = nn.Sequential(self.scale_in, self.msd, self.scale_out)
        self.net.cuda()

        # Train all parameters apart from self.scale_in.
        # self.init_optimizer(self.msd)

    def __call__(self, input_):
        return self.net(input_)
        
    def eval(self):
        self.msd.eval()

    def train(self):
        self.msd.train()
