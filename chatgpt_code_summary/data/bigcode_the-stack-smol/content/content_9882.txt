import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
import torch.nn.functional as F
import math
import torch.utils.model_zoo as model_zoo

nonlinearity = nn.ReLU
class EncoderBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride):
        super().__init__()
        
        self.c1=nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=False)
        self.bn1=nn.BatchNorm2d(outchannel)
        self.re1=nn.ReLU(inplace=True)
        self.c2=nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=False)
        self.bn2=nn.BatchNorm2d(outchannel)
        self.re2=nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.c1(x)
        x = self.bn1(x)
        x = self.re1(x)
        x = self.c2(x)
        x = self.bn2(x)
        x = self.re2(x)
        return x
class EncoderBlock0(nn.Module):
    def __init__(self, in_channels, n_filters):
        super().__init__()

        # B, C, H, W -> B, C/4, H, W
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity(inplace=True)

        # B, C/4, H, W -> B, C/4, H, W
        self.pool = nn.MaxPool2d(2, 2)#
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity(inplace=True)

        # B, C/4, H, W -> B, C, H, W
        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.pool(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super().__init__()

        # B, C, H, W -> B, C/4, H, W
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity(inplace=True)

        # B, C/4, H, W -> B, C/4, H, W
        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3,
                                          stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity(inplace=True)

        # B, C/4, H, W -> B, C, H, W
        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x

class ChannelSE(nn.Module):
    def __init__(self,inchannel):
        super().__init__()
        self.lin1=torch.nn.Linear(inchannel, inchannel//2)
        self.lin2=torch.nn.Linear(inchannel//2, inchannel)
        self.c=inchannel
    def forward(self,x):
        #_,c,h,w=x.size
        #print(c)
        #print(h)
        #print(w)
        m=torch.mean(torch.mean(x,dim=2,keepdim=True),dim=3,keepdim=True)
        m = m.view(m.size(0), -1)
        m=self.lin1(m)
        m=nn.ReLU()(m)
        m=self.lin2(m)

        m=nn.Sigmoid()(m)
        m = m.view(m.size(0), self.c,1,1)
        x=m*x#torch.matmul(m,x)
        return x

class SpatialSE(nn.Module):
    def __init__(self,inchannel):
        super().__init__()
        self.conv=torch.nn.Conv2d(inchannel,1,kernel_size=1,stride=1)
    def forward(self,x):
        #_,c,h,w=x.size
        #print(c)
        #print(h)
        #print(w)
        m = self.conv(x)
        m=nn.Sigmoid()(m)
        x=m*x#torch.matmul(m,x)
        return x

class DecoderBlockv(nn.Module):
    def __init__(self, in_channels, n_filters):
        super().__init__()

        # B, C, H, W -> B, C/4, H, W
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity(inplace=True)

        # B, C/4, H, W -> B, C/4, H, W
        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3,
                                          stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity(inplace=True)

        # B, C/4, H, W -> B, C, H, W
        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity(inplace=True)
        
        self.cSE = ChannelSE(n_filters)
        self.sSE = SpatialSE(n_filters)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        x = self.cSE(x) + self.sSE(x)
        return x
class ConvUp(nn.Module):
    def __init__(self, in_channels, n_filters):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2,mode='bilinear')
        self.conv1 = nn.Conv2d(in_channels, n_filters, 3, padding = 1)
        self.norm1 = nn.BatchNorm2d(n_filters)
        self.relu1 = nonlinearity(inplace=True)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        return x   
class ConscSE(nn.Module):
    def __init__(self, n_filters):
        super().__init__()

        
        self.cSE = ChannelSE(n_filters)
        self.sSE = SpatialSE(n_filters)
        
    def forward(self, x):
        x = self.cSE(x) + self.sSE(x)
        return x
    
class DecoderBlockup(nn.Module):
    def __init__(self, in_channels, n_filters):
        super().__init__()

        # B, C, H, W -> B, C/4, H, W
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity(inplace=True)

        # B, C/4, H, W -> B, C/4, H, W
        self.deconv2 = ConvUp(in_channels // 4, in_channels // 4)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity(inplace=True)

        # B, C/4, H, W -> B, C, H, W
        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity(inplace=True)
        
        self.cSE = ChannelSE(n_filters)
        self.sSE = SpatialSE(n_filters)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        x = self.cSE(x) + self.sSE(x)
        return x
    
class DecoderBlock23(nn.Module):
    def __init__(self, in_channels, n_filters, scal=4):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2,mode='bilinear')
        # B, C, H, W -> B, C/4, H, W
        self.conv1 = nn.Conv2d(in_channels, in_channels // scal, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // scal)
        self.relu1 = nonlinearity(inplace=True)

        # B, C/4, H, W -> B, C, H, W
        self.conv2 = nn.Conv2d(in_channels // scal, n_filters, 1)
        self.norm2 = nn.BatchNorm2d(n_filters)
        self.relu2 = nonlinearity(inplace=True)
        
        self.cSE = ChannelSE(n_filters)
        self.sSE = SpatialSE(n_filters)
        
    def forward(self, x):
        x = self.up(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        #x = self.cSE(x) + self.sSE(x)
        return x
    
class Upscale:
    transposed_conv = 0
    upsample_bilinear = 1
    pixel_shuffle = 2  
class BasicDecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, conv_size=3, upscale=Upscale.transposed_conv):
        super().__init__()
        padding = 0
        if conv_size == 3:
            padding = 1
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, conv_size, padding=padding),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True)
        )

        last_conv_channels = middle_channels
        if upscale == Upscale.transposed_conv:
            self.layer2 = nn.Sequential(
                nn.ConvTranspose2d(middle_channels, middle_channels, 3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(middle_channels),
                nn.ReLU(inplace=True)
            )
        elif upscale == Upscale.upsample_bilinear:
            self.layer2 = nn.Upsample(scale_factor=2)
        else:
            self.layer2 = nn.PixelShuffle(upscale_factor=2)
            last_conv_channels = middle_channels // 4

        self.layer3 = nn.Sequential(
            nn.Conv2d(last_conv_channels, out_channels, conv_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

class UnetBNDecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, upscale=Upscale.upsample_bilinear):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layer(x)
    
class LinkNet34a(nn.Module):
    def __init__(self, num_classes, num_channels=3):
        super().__init__()
        assert num_channels == 3, "num channels not used now. to use changle first conv layer to support num channels other then 3"
        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        # Center
        self.center = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(filters[3], filters[1], 3, padding=1),
            nn.BatchNorm2d(filters[1]),
            nn.ReLU(inplace=True)            
        
        )
        # Decoder
        self.decoder5 = UnetBNDecoderBlock(filters[1],filters[2]//4, filters[2])#
        self.conv5=nn.Conv2d(256+512,256,1) 
        self.decoder4 = UnetBNDecoderBlock(filters[2],filters[2]//4, filters[1])#DecoderBlock(filters[3], filters[2])
        self.conv4=nn.Conv2d(128+256,256,1)        
        self.decoder3 = UnetBNDecoderBlock(filters[2],filters[2]//4, filters[0])#DecoderBlock(filters[2], filters[1])
        self.conv3=nn.Conv2d(64+128,128,1)        
        self.decoder2 = UnetBNDecoderBlock(filters[1],filters[1]//4, filters[0])#DecoderBlock(filters[1], filters[0])
        self.conv2=nn.Conv2d(128,64,1)        
        #self.decoder1 = UnetBNDecoderBlock(filters[0],filters[0]//4, filters[0])#DecoderBlock(filters[0], filters[0])

        # Final Classifier
        self.finaldeconv1 = UnetBNDecoderBlock(filters[0],filters[0]//4, filters[0])#ConvUp(filters[0], filters[0])
        
        # Final Classifier
        self.logit = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nonlinearity(inplace=True),
            nn.Conv2d(64, 1, 1),
        )        


    # noinspection PyCallingNonCallable
    def forward(self, x):
        # Encoder
        x = x.float()
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        #x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        ############################
        e5 = self.center(e4)
        d5 = torch.cat([self.decoder5(e5) , e4], 1)#concat([self.decoder5(e5) , e4])
        d5 = self.conv5(d5)     
        #########################
        d4 = torch.cat([self.decoder4(d5) , e3], 1)#concat([self.decoder5(e5) , e4])
        d4 = self.conv4(d4)
        # d4 = e3
        #d3 = self.decoder3(d4) + e2
        #print(e2.shape)
        d3 = torch.cat([self.decoder3(d4) , e2], 1)#concat([self.decoder5(e5) , e4])
        #print(d3.shape)
        d3 = self.conv3(d3)
        
        #d2 = self.decoder2(d3) + e1
        d2 = torch.cat([self.decoder2(d3) , e1], 1)#concat([self.decoder5(e5) , e4])
        d2 = self.conv2(d2)
        
        #d1 = self.decoder1(d2)

        # Final Classification
        f = self.finaldeconv1(d2)
        #f = self.finalrelu1(f)
        f = self.logit(f)
        return f


class DecoderBlockH(nn.Module):
    def __init__(self, in_channels,channels, n_filters):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2,mode='bilinear')
        
        # B, C, H, W -> B, C/4, H, W
        self.conv1 = nn.Conv2d(in_channels, channels, 3, padding=1)
        self.norm1 = nn.BatchNorm2d(channels)
        self.relu1 = nonlinearity(inplace=True)

        # B, C/4, H, W -> B, C, H, W
        self.conv2 = nn.Conv2d(channels, n_filters, 3, padding=1)
        self.norm2 = nn.BatchNorm2d(n_filters)
        self.relu2 = nonlinearity(inplace=True)
        
        #self.cSE = ChannelSE(n_filters)
        #self.sSE = SpatialSE(n_filters)
        
    def forward(self, x, e=None):
        x = self.up(x)
        if e is not None:
            x = torch.cat([x, e], 1)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        #x = self.cSE(x) + self.sSE(x)
        return x


class ConvBn2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.layer = nn.Sequential(
            #nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_channels, out_channels, kernel_size, padding),
            nn.BatchNorm2d(out_channels),
            #nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layer(x)
    
class Decoder3(nn.Module):
    def __init__(self, in_channels,res_channels, channels, n_filters):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        
        # B, C, H, W -> B, C/4, H, W
        self.conv1 = nn.Conv2d(in_channels+res_channels, channels, 3, padding=1)
        self.norm1 = nn.BatchNorm2d(channels)
        self.relu1 = nonlinearity(inplace=True)

        # B, C/4, H, W -> B, C, H, W
        self.conv2 = nn.Conv2d(channels, n_filters, 3, padding=1)
        self.norm2 = nn.BatchNorm2d(n_filters)
        self.relu2 = nonlinearity(inplace=True)
        
        self.SCSE = SCSEBlock(n_filters)#ChannelSE(n_filters)
        #self.sSE = SpatialSE(n_filters)
        
    def forward(self, x, e=None):
        x = self.up(x)
        if e is not None:
            x = torch.cat([x, e], 1)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.SCSE(x)# + self.sSE(x)
        return x
    
class DenseNet34(nn.Module):
    def __init__(self ):
        super().__init__()
        #super(Net,self).__init__()
        filters = [64, 128, 256, 512]
        self.resnet =  models.resnet34(pretrained=True)#ResNet(BasicBlock, [3, 4, 6, 3], num_classes=1 )

        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.encoder2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            self.resnet.layer1,
        )
        self.encoder3 = self.resnet.layer2
        self.encoder4 = self.resnet.layer3
        self.encoder5 = self.resnet.layer4

        self.center = nn.Sequential(
            ConvBn2d( 512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ConvBn2d( 256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
######################################################################
        #self.decoder5 = Decoder3(256, 512, 512, 64)
        #self.decoder4 = Decoder3( 64, 256, 256, 64)
        #self.decoder3 = Decoder3( 64, 128, 128, 64)
        #self.decoder2 = Decoder3( 64,  64,  64, 64)
        #self.decoder1 = Decoder3( 64,  64,  32, 64)
        
        self.decoder5 = DecoderBlockH(filters[3]+filters[2],filters[2], 64)
        #self.conv5=nn.Conv2d(64+512,64,1)#before or after SE?
        self.se5=SCSEBlock(64)
        self.decoder4 = DecoderBlockH(filters[2]+64, filters[1], 64)
        #self.conv4=nn.Conv2d(64+256,64,1)        
        self.se4=SCSEBlock(64)
        self.decoder3 = DecoderBlockH(filters[1]+64, filters[1], 64)
        #self.conv3=nn.Conv2d(64+128,64,1)    
        self.se3=SCSEBlock(64)
        self.decoder2 = DecoderBlockH(filters[0]+64, filters[0], 64)
        #self.conv2=nn.Conv2d(64+64,64,1)   
        self.se2=SCSEBlock(64)
        self.decoder1 = DecoderBlockH(filters[0], filters[0]//2, 64)
        self.se1=SCSEBlock(64)        
        
##############################################################################        
        self.fuse_pixel  = nn.Sequential(        
            nn.Conv2d(320, 64, kernel_size=3, padding=1),
        )
        self.logit_pixel  = nn.Sequential(
            #nn.Conv2d(320, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d( 64,  1, kernel_size=1, padding=0),
        )

        self.logit_image = nn.Sequential(
            #nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            #nn.Linear(128, 1),
            nn.Linear(64, 1),
        )
        self.fuse_image = nn.Sequential(
            nn.Linear(512, 64),
            #nn.ReLU(inplace=True),
            #nn.Linear(128, 1),
        )
        self.fuse = nn.Sequential(
            #nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            #nn.Conv2d(128, 64, kernel_size=1, padding=0),
            #nn.BatchNorm2d(64),
            #nn.ReLU(inplace=True),
        )
        self.logit = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=1, padding=0),
            #nn.ReLU(inplace=True),
            #nn.Conv2d( 64,  1, kernel_size=1, padding=0),
        )

    def forward(self, x):
        batch_size,C,H,W = x.shape
        """
        mean=[0.485, 0.456, 0.406]
        std =[0.229, 0.224, 0.225]
        x = torch.cat([
            (x-mean[2])/std[2],
            (x-mean[1])/std[1],
            (x-mean[0])/std[0],
        ],1)
        """
        x = x.float()

        e1 = self.encoder1(x )  #; print('e1',e1.size())
        e2 = self.encoder2(e1)  #; print('e2',e2.size())
        e3 = self.encoder3(e2)  #; print('e3',e3.size())
        e4 = self.encoder4(e3)  #; print('e4',e4.size())
        e5 = self.encoder5(e4)  #; print('e5',e5.size())

        f = self.center(e5)                #; print('f',f.size())
        #print(f.shape)
        #print(e5.shape)
        #e1 = self.encoder1(x)#
        #e2 = self.encoder2(e1)#
        #e3 = self.encoder3(e2)#
        #e4 = self.encoder4(e3)#

        #e5 = self.center(e4)#512
        ####################################################################################
        #d5 = self.decoder5( f,e5)          #; print('d5',f.size())
        #d4 = self.decoder4(d5,e4)          #; print('d4',f.size())
        #d3 = self.decoder3(d4,e3)          #; print('d3',f.size())
        #d2 = self.decoder2(d3,e2)          #; print('d2',f.size())
        #d1 = self.decoder1(d2,e1)          #; print('d1',f.size())
        
        d5 = self.decoder5(f,e5)
        d5 = self.se5(d5)
        # Decoder with Skip Connections
        #d4 = self.decoder4(d5) + e3
        #d4 = torch.cat([self.decoder4(d5) , e3], 1)#concat([self.decoder5(e5) , e4])
        #print(d5.shape)
        #print(e3.shape)
        d4 = self.decoder4(d5,e4)
        d4 = self.se4(d4)
        # d4 = e3
        #d3 = self.decoder3(d4) + e2
        #print(e2.shape)
        #d3 = torch.cat([self.decoder3(d4) , e2], 1)#concat([self.decoder5(e5) , e4])
        #print(d3.shape)
        d3 = self.decoder3(d4,e3)
        d3 = self.se3(d3)
        
        #d2 = self.decoder2(d3) + e1
        #d2 = torch.cat([self.decoder2(d3) , e1], 1)#concat([self.decoder5(e5) , e4])
        d2 = self.decoder2(d3,e2)
        d2 = self.se2(d2)
        d1 = self.decoder1(d2)
        d1 = self.se1(d1)
        ########################################################################################
        d = torch.cat((
            d1,
            F.upsample(d2,scale_factor= 2, mode='bilinear',align_corners=False),
            F.upsample(d3,scale_factor= 4, mode='bilinear',align_corners=False),
            F.upsample(d4,scale_factor= 8, mode='bilinear',align_corners=False),
            F.upsample(d5,scale_factor=16, mode='bilinear',align_corners=False),
        ),1)
        
        #######################################################################
        """
        d = F.dropout(d, p=0.50, training=self.training)
        logit_pixel = self.logit_pixel(d)

        f = F.adaptive_avg_pool2d(e5, output_size=1).view(batch_size,-1)
        f = F.dropout(f, p=0.50, training=self.training)
        logit_image = self.logit_image(f).view(-1)
        """
        ###########################################################################
        #d = torch.cat([d1,d2,d3,d4,d5],1) #hyper-columns
        d = F.dropout(d, p=0.50, training=self.training)
        fuse_pixel  = self.fuse_pixel(d)#64-128-128
        logit_pixel = self.logit_pixel(fuse_pixel)#1-128-128

        e = F.adaptive_avg_pool2d(e5, output_size=1).view(batch_size,-1) #image pool#-512-1-1
        e = F.dropout(e, p=0.50, training=self.training)#
        fuse_image  = self.fuse_image(e)#-64-1-1
        logit_image = self.logit_image(fuse_image).view(-1)#-1-1-1

        #fuse = self.fuse(torch.mul(fuse_pixel, F.upsample(fuse_image.view(batch_size,-1,1,1,),scale_factor=128, mode='nearest')))
        #fuse = self.fuse(fuse_pixel+ F.upsample(fuse_image.view(batch_size,-1,1,1,),scale_factor=128, mode='nearest'))
        fuse = self.fuse(torch.cat([ #fuse
            fuse_pixel,
            F.upsample(fuse_image.view(batch_size,-1,1,1,),scale_factor=128, mode='nearest')
        ],1))
        logit = self.logit(fuse)#1-128-128

        return logit, logit_pixel, logit_image        
        
        
        #return logit_pixel, logit_image


    ##-----------------------------------------------------------------


    #def criterion(self, logit_pixel, logit_image, truth_pixel, truth_image, is_average=True):

    
    
    
"""
d3 = F.upsample(d3,scale_factor= 4, mode='bilinear',align_corners=False)
    d4 = F.upsample(d4,scale_factor= 8, mode='bilinear',align_corners=False)
    d5 = F.upsample(d5,scale_factor=16, mode='bilinear',align_corners=False)

    d = torch.cat([d1,d2,d3,d4,d5],1) #hyper-columns
    d = F.dropout(d, p=0.50, training=self.training)
    fuse_pixel  = self.fuse_pixel(d)
    logit_pixel = self.logit_pixel(fuse_pixel)

    e = F.adaptive_avg_pool2d(e5, output_size=1).view(batch_size,-1) #image pool
    e = F.dropout(e, p=0.50, training=self.training)
    fuse_image  = self.fuse_image(e)
    logit_image = self.logit_image(fuse_image).view(-1)

    fuse = self.fuse(torch.cat([ #fuse
        fuse_pixel,
        F.upsample(fuse_image.view(batch_size,-1,1,1,),scale_factor=128, mode='nearest')
    ],1))
    logit = self.logit(fuse)

    return logit, logit_pixel, logit_image
"""


