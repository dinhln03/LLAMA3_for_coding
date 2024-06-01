import torch
import torch.nn as nn

from .subnet import DoubleConv, UpConv, RRCU


class R2UNet(nn.Module):
    def __init__(self, in_ch, out_ch, base_ch=64):
        super(R2UNet, self).__init__()
        self.inc = DoubleConv(in_ch, base_ch)
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down1 = self._Down(base_ch, base_ch * 2)
        self.down2 = self._Down(base_ch * 2, base_ch * 4)
        self.down3 = self._Down(base_ch * 4, base_ch * 8)
        self.down4 = self._Down(base_ch * 8, base_ch * 16)
        self.up1 = self._Up(base_ch * 16, base_ch * 8)
        self.up2 = self._Up(base_ch * 8, base_ch * 4)
        self.up3 = self._Up(base_ch * 4, base_ch * 2)
        self.up4 = self._Up(base_ch * 2, base_ch)
        self.outc = nn.Conv2d(base_ch, out_ch, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
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
        return x
    
    class _Down(nn.Module):
        def __init__(self, in_ch, out_ch):
            super(R2UNet._Down, self).__init__()
            self.down = nn.Sequential(
                nn.MaxPool2d(kernel_size=2),
                RRCU(in_ch, out_ch)
            )
        
        def forward(self, x):
            x = self.down(x)
            return x
    
    class _Up(nn.Module):
        def __init__(self, in_ch, out_ch):
            super(R2UNet._Up, self).__init__()
            self.up = UpConv(in_ch, out_ch)
            self.conv = RRCU(in_ch, out_ch)
        
        def forward(self, x1, x2):
            x1 = self.up(x1)
            x = torch.cat([x2, x1], dim=1)
            x = self.conv(x)
            return x
