import torch.nn as nn
import torch.nn.functional as F
import torch

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(mid_channels),
            nn.GroupNorm(num_groups=1, num_channels=mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(out_channels),
            nn.GroupNorm(num_groups=1, num_channels=out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    
class OnlyUp(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1):
        x1 = self.up(x1)
        return self.conv(x1)


class UnetGenerator(nn.Module):
    def __init__(self, in_channels, out_channels, n_features = 24):
        super(UnetGenerator, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.inc11 = DoubleConv(in_channels, n_features)
        self.down11 = Down(n_features, n_features * 2)
        self.down12 = Down(n_features * 2, n_features * 4)
        self.down13 = Down(n_features * 4, n_features * 4)
        self.inc12 = DoubleConv(n_features * 4, n_features * 4)
        self.up11 = Up(n_features * 8, n_features * 2)
        self.up12 = Up(n_features * 4, n_features)
        self.up13 = OnlyUp(n_features, n_features)
        
        self.down21 = Down(n_features, n_features * 2)
        self.down22 = Down(n_features * 2, n_features * 4)
        self.down23 = Down(n_features * 4, n_features * 4)
        self.inc2 = DoubleConv(n_features * 4, n_features * 4)
        self.up21 = Up(n_features * 8, n_features * 2)
        self.up22 = Up(n_features * 4, n_features)
        self.up23 = OnlyUp(n_features, n_features)
        
        self.down31 = Down(n_features, n_features * 2)
        self.down32 = Down(n_features * 2, n_features * 4)
        self.down33 = Down(n_features * 4, n_features * 4)
        self.inc3 = DoubleConv(n_features * 4, n_features * 4)
        self.up31 = Up(n_features * 8, n_features * 2)
        self.up32 = Up(n_features * 4, n_features)
        self.up33 = OnlyUp(n_features, n_features)
        
        self.down41 = Down(n_features, n_features * 2)
        self.down42 = Down(n_features * 2, n_features * 4)
        self.down43 = Down(n_features * 4, n_features * 4)
        self.inc4 = DoubleConv(n_features * 4, n_features * 4)
        self.up41 = Up(n_features * 8, n_features * 2)
        self.up42 = Up(n_features * 4, n_features)
        self.up43 = OnlyUp(n_features, n_features)
        
        self.down51 = Down(n_features, n_features * 2)
        self.down52 = Down(n_features * 2, n_features * 4)
        self.down53 = Down(n_features * 4, n_features * 4)
        self.inc5 = DoubleConv(n_features * 4, n_features * 4)
        self.up51 = Up(n_features * 8, n_features * 2)
        self.up52 = Up(n_features * 4, n_features)
        self.up53 = OnlyUp(n_features, n_features)
        
        self.down61 = Down(n_features, n_features * 2)
        self.down62 = Down(n_features * 2, n_features * 4)
        self.down63 = Down(n_features * 4, n_features * 4)
        self.inc6 = DoubleConv(n_features * 4, n_features * 4)
        self.up61 = Up(n_features * 8, n_features * 2)
        self.up62 = Up(n_features * 4, n_features)
        self.up63 = OnlyUp(n_features, out_channels)

    def forward(self, x):
        x_inc11 = self.inc11(x)
        x11 = self.down11(x_inc11)
        x12 = self.down12(x11)
        x13 = self.down13(x12)
        x_inc12 = self.inc12(x13)
        x1 = self.up11(x_inc12, x12)
        x1 = self.up12(x1, x11)
        x1 = self.up13(x1)
        
        x21 = self.down21(x1)
        x22 = self.down22(x21)
        x23 = self.down23(x22)
        x_inc2 = self.inc2(x23)
        x2 = self.up21(x_inc2, x22)
        x2 = self.up22(x2, x21)
        x2 = self.up23(x2)
        
        x31 = self.down31(x2)
        x32 = self.down32(x31)
        x33 = self.down33(x32)
        x_inc3 = self.inc3(x33)
        x3 = self.up31(x_inc3, x32)
        x3 = self.up32(x3, x31)
        x3 = self.up33(x3)
        
        x41 = self.down41(x3)
        x42 = self.down42(x41)
        x43 = self.down43(x42)
        x_inc4 = self.inc4(x43)
        x4 = self.up41(x_inc4, x42)
        x4 = self.up42(x4, x41)
        x4 = self.up43(x4)
        
        x51 = self.down51(x4)
        x52 = self.down52(x51)
        x53 = self.down53(x52)
        x_inc5 = self.inc5(x53)
        x5 = self.up51(x_inc5, x52)
        x5 = self.up52(x5, x51)
        x5 = self.up53(x5)
        
        x61 = self.down61(x5)
        x62 = self.down62(x61)
        x63 = self.down63(x62)
        x_inc6 = self.inc6(x63)
        x6 = self.up61(x_inc6, x62)
        x6 = self.up62(x6, x61)
        logits = self.up63(x6)
        
        return logits
