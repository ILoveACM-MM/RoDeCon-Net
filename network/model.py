import math
import torch
import torch.nn as nn
from network.resnet import resnet50
from network.pvtv2 import pvt_v2_b2
import torch.nn.functional as F


import torch

import torch.nn as nn


import torchvision

class DirectionOffsets(nn.Module):
    def __init__(self, in_channels, kernel_size=3):
        super().__init__()
        
        self.offset1=nn.Sequential(
            nn.Conv2d(in_channels,in_channels,(1,15),1,(0,7),groups=in_channels),
            nn.BatchNorm2d(in_channels)
        )
        self.offset2=nn.Sequential(
            nn.Conv2d(in_channels,in_channels,(15,1),padding=(7,0),groups=in_channels),
            nn.BatchNorm2d(in_channels)
        )
        self.offset3=nn.Sequential(
            nn.Conv2d(in_channels,in_channels,3,padding=1,groups=in_channels),
            nn.BatchNorm2d(in_channels)
        )
        self.balance=nn.Sequential(
            nn.Conv2d(in_channels,2 * kernel_size * kernel_size,1),
            nn.BatchNorm2d(2 * kernel_size * kernel_size)
        )
        
    def forward(self, x):
        B,C,H,W=x.shape
        offsets1=self.offset1(x)
        offsets2=self.offset2(x)
        offsets3=self.offset3(x)
        offsets=offsets1+offsets2+offsets3
        offsets=self.balance(offsets)
        return offsets

class Deform(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        
        self.offset=DirectionOffsets(in_channels)
        
        self.deform2=torchvision.ops.DeformConv2d(in_channels=in_channels,
                                                        out_channels=in_channels,
                                                        kernel_size=3,
                                                        padding=2,
                                                        groups=in_channels,
                                                        dilation=2,
                                                        bias=False)
        
        self.deform3=torchvision.ops.DeformConv2d(in_channels=in_channels,
                                                        out_channels=in_channels,
                                                        kernel_size=3,
                                                        padding=3,
                                                        groups=in_channels,
                                                        dilation=3,
                                                        bias=False)
        
        self.deform4=torchvision.ops.DeformConv2d(in_channels=in_channels,
                                                        out_channels=in_channels,
                                                        kernel_size=3,
                                                        padding=4,
                                                        groups=in_channels,
                                                        dilation=4,
                                                        bias=False)
        
        self.deform5=torchvision.ops.DeformConv2d(in_channels=in_channels,
                                                        out_channels=in_channels,
                                                        kernel_size=3,
                                                        padding=5,
                                                        groups=in_channels,
                                                        dilation=5,
                                                        bias=False)
        
        self.balance=nn.Sequential(
            nn.Conv2d(in_channels,in_channels,1),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x):
        offsets = self.offset(x)
        out = self.deform2(x, offsets)+self.deform3(x, offsets)+self.deform4(x, offsets)+self.deform5(x, offsets)
        out=self.balance(out)*x
        return out


class CBR(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, stride=1, act=True,groups=1):
        super().__init__()
        self.act = act
        if in_c<out_c:
            groups=in_c
        elif in_c>out_c:
            groups=out_c
        else:
            groups=out_c
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, padding=padding, dilation=dilation, bias=False, stride=stride,groups=groups),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)
        self.balance=nn.Conv2d(out_c, out_c, 1)

    def forward(self, x):
        x = self.conv(x)
        if self.act == True:
            x = self.relu(x)
        x=self.balance(x)
        return x



class FeatureDecoupling(nn.Module):
    def __init__(self, in_channle=1024, out_channel=1024):
        super().__init__()
        '''
        Adopting three branches to decompose the feature map into foreground, background, and uncertain regions.
        '''
        #foreground branch
        self.cbr_fg = nn.Sequential(
            CBR(in_channle, in_channle//2, kernel_size=3, padding=1),
            CBR(in_channle//2, out_channel, kernel_size=3, padding=1),
            CBR(out_channel, out_channel, kernel_size=1, padding=0)
        )
        #background branch
        self.cbr_bg = nn.Sequential(
            CBR(in_channle, in_channle//2, kernel_size=3, padding=1),
            CBR(in_channle//2, out_channel, kernel_size=3, padding=1),
            CBR(out_channel, out_channel, kernel_size=1, padding=0)
        )
        #uncertain branch
        self.cbr_uc = nn.Sequential(
            CBR(out_channel, in_channle//2, kernel_size=3, padding=1),
            CBR(in_channle//2, out_channel, kernel_size=3, padding=1),
            CBR(out_channel, out_channel, kernel_size=1, padding=0)
        )

    def forward(self, x):
        #positive attention
        xf=x.sigmoid()
        #inverse attention
        xb=1-x.sigmoid()
        #foreground branch
        f_fg = self.cbr_fg(xf)
        #background branch
        f_bg = self.cbr_bg(xb)
        #modualate the uncertain regions
        xu=torch.exp(-torch.abs(f_fg-f_bg))
        #uncertain branch
        f_uc = self.cbr_uc(xu)
        return f_fg, f_bg, f_uc



class FeatureDecouplingPredictionHead(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        '''
        generating foreground, background, and uncertain masks.
        '''
        #generating foreground masks
        self.branch_fg = nn.Sequential(
            CBR(in_channel, in_channel//4, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1/8
            CBR(in_channel//4, in_channel//4, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1/4
            CBR(in_channel//4, in_channel//8, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1/2
            CBR(in_channel//8, in_channel//16, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1
            CBR(in_channel//16, in_channel//16, kernel_size=3, padding=1),
            nn.Conv2d(in_channel//16, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        #generating background masks
        self.branch_bg = nn.Sequential(
            CBR(in_channel, in_channel//4, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1/8
            CBR(in_channel//4, in_channel//4, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1/4
            CBR(in_channel//4, in_channel//8, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1/2
            CBR(in_channel//8, in_channel//16, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1
            CBR(in_channel//16, in_channel//16, kernel_size=3, padding=1),
            nn.Conv2d(in_channel//16, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        #generating uncertain masks
        self.branch_uc = nn.Sequential(
            CBR(in_channel, in_channel//4, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1/8
            CBR(in_channel//4, in_channel//4, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1/4
            CBR(in_channel//4, in_channel//8, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1/2
            CBR(in_channel//8, in_channel//16, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1
            CBR(in_channel//16, in_channel//16, kernel_size=3, padding=1),
            nn.Conv2d(in_channel//16, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, f_fg, f_bg, f_uc):
        '''
        input dimension: (B,C,H,W)
        output dimension: (B,1,H,W)
        '''
        #generating foreground masks
        mask_fg = self.branch_fg(f_fg)
        #generating background masks
        mask_bg = self.branch_bg(f_bg)
        #generating uncertain masks
        mask_uc = self.branch_uc(f_uc)
        
        return mask_fg, mask_bg, mask_uc


class ContrastDrivenFeatureAlignment(nn.Module):
    def __init__(self, in_channel, out_channel, window_size=3, window_padding=1, window_stride=1, attn_drop=0., proj_drop=0.):
        super().__init__()
        '''
        num_heads : the multi-head number
        window_size : the size of local attention window
        '''
        self.window_size = window_size
        self.window_padding = window_padding
        self.window_stride = window_stride
        self.scale = out_channel ** -0.5
        self.dim=out_channel

        #adjust the shape of foreground and background feature maps
        
        self.pro_x=nn.Conv2d(in_channel,out_channel,kernel_size=1)
        self.pro_b=nn.Conv2d(out_channel,out_channel,kernel_size=1)
        self.pro_f=nn.Conv2d(out_channel,out_channel,kernel_size=1)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        
        #perform unfold operation for encoding feature
        self.unfold = nn.Unfold(kernel_size=window_size, padding=window_padding, stride=window_stride)
        # self.pool = nn.AvgPool2d(kernel_size=window_stride, stride=window_stride, ceil_mode=True)

        #Fusion operation
        self.output_cbr = nn.Sequential(
            nn.Conv2d(2*out_channel, out_channel, kernel_size=3, padding=1,groups=out_channel),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=1),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1,groups=out_channel),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=1),
        )
        
    def forward_collect_value(self,x,fg,bg,B):
        #collect neighbor value in K*K window
        
        x_unfolded = self.unfold(x).reshape(B, -1,self.window_size,self.window_size)
        f_unfolded = self.unfold(fg).reshape(B, -1,self.window_size,self.window_size)
        b_unfolded = self.unfold(bg).reshape(B, -1,self.window_size,self.window_size)
        x_unfolded=self.proj_drop(x_unfolded)
        f_unfolded=self.proj_drop(f_unfolded)
        b_unfolded=self.proj_drop(b_unfolded)
        return x_unfolded,f_unfolded,b_unfolded
    
    def forward_optimize_fb(self,x_unfolded,f_unfolded,b_unfolded):
        #window conv
        # x_unfolded=self.window_conv_x(x_unfolded)
        # f_unfolded=self.window_conv_f(f_unfolded)
        # b_unfolded=self.window_conv_b(b_unfolded)
        #optimizing foreground
        f_x=x_unfolded*f_unfolded
        #suppressing background
        b_x=x_unfolded-b_unfolded
        return f_x,b_x
        
    def forward_compute_attention(self,f_x,b_x,x_unfolded,f_unfolded,b_unfolded):
        #calculating foreground and background window attention
        # window_sum_f = f_x.sum(dim=(-2, -1), keepdim=True)
        # window_sum_b = b_x.sum(dim=(-2, -1), keepdim=True)
        # attn_f_x = f_x / window_sum_f
        # attn_b_x = b_x / window_sum_b
        attn_f_x=f_x.softmax(dim=-1)
        attn_b_x=b_x.softmax(dim=-1)
        #calculating comprehensive window attention
        attn_f=attn_f_x*x_unfolded*self.scale
        attn_b=attn_b_x*x_unfolded*self.scale
        #Guiding the model via attention
        attn_f=self.attn_drop(attn_f)
        attn_b=self.attn_drop(attn_b)
        
        return attn_f,attn_b
    
        
    def forward(self, x, fg, bg):
        '''
        input shape: (B,C,H,W)
        '''
        B, C,H, W= x.shape
        #projection
        x = self.pro_x(x) #(B,C,H,W)
        fg = self.pro_f(fg) #(B,C,H,W)
        bg = self.pro_b(bg) #(B,C,H,W)
     
        #collect neighbor values
        x_unfolded,f_unfolded,b_unfolded = self.forward_collect_value(x,fg,bg,B) #(B,dc,K,K)
      
        #optimizing foreground, and suppressing background
        f_x,b_x=self.forward_optimize_fb(x_unfolded,f_unfolded,b_unfolded) #(B,dc,K,K)
      
        #calculating window attention
        attn_f,attn_b=self.forward_compute_attention(f_x,b_x,x_unfolded,f_unfolded,b_unfolded) #(B,dc,K,K)
        
        out=self.forward_integrating_values(attn_f,attn_b,B,C,H,W)
        return out

    def forward_integrating_values(self,attn_f,attn_b,B,C,H,W):
        '''
        input shape:B,dc,K,K
        output shape: B,C,H,W
        '''
        #B,dc,K,K -> B,C*K*K,H*W
        
        attn_f=attn_f.permute(0,2,3,1).contiguous().view(B,self.window_size*self.window_size*self.dim,-1)
        attn_b=attn_b.permute(0,2,3,1).contiguous().view(B,self.window_size*self.window_size*self.dim,-1)
        #integrating neighbor values in K*K window
        attn_f=F.fold(attn_f, output_size=(H, W), kernel_size=self.window_size,
                            padding=self.window_padding, stride=self.window_stride)
        attn_b=F.fold(attn_b, output_size=(H, W), kernel_size=self.window_size,
                            padding=self.window_padding, stride=self.window_stride)
        attn=torch.cat([attn_f,attn_b],dim=1)
        attn = self.proj_drop(attn)#B,C,H,W
        attn=self.output_cbr(attn)
        return attn



class CrossLevelFeatureCascade(nn.Module):
    def __init__(self, in_channel, hidden_channel,out_channel,scale=2):
        super().__init__()
        self.scale = scale        
        self.relu = nn.ReLU(inplace=True)
        #projection
        self.dim=CBR(in_channel , out_channel, kernel_size=1, padding=0)
        #upsample and projection
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=scale, mode="bilinear", align_corners=True),
            CBR(hidden_channel , out_channel, kernel_size=1, padding=0)
        )
        #multi-scale deforming conv
        self.deform =Deform(out_channel)        
        
        self.c = CBR(out_channel, out_channel, act=False)

    def forward(self, x, skip):
        #projection
        x=self.dim(x)
        #upsample and projection
        skip = self.up(skip)
        x = x+skip
        x=self.c(x)+x
        #multi-scale deforming conv
        x=self.deform(x)
        return x



class CrossLevelFeatureCascadeUnit(nn.Module):
    def __init__(self,in_channel=[64,256,512,1024],final_channel=128,scale=2):
        super().__init__()

        """ cross-level feature cascade and multi-scale deforming feature fusion  """
        self.cfcu_small = CrossLevelFeatureCascade(in_channel=in_channel[0],hidden_channel=in_channel[1],out_channel=final_channel, scale=scale)
        self.cfcu_middle = CrossLevelFeatureCascade(in_channel=in_channel[1],hidden_channel=in_channel[2], out_channel=final_channel, scale=scale)
        self.cfcu_large = CrossLevelFeatureCascade(in_channel=in_channel[2],hidden_channel=in_channel[3], out_channel=final_channel, scale=scale)

    def forward(self, f1, f2, f3, f4):
        """ cross-level feature cascade and multi-scale deforming feature fusion  """
        f_small = self.cfcu_small(f1, f2)
        f_middle = self.cfcu_middle(f2, f3)
        f_large = self.cfcu_large(f3, f4)

        return f_small, f_middle, f_large

class FinalPredictionHead(nn.Module):

    def __init__(self, in_channel, out_channel=1):
        super().__init__()
        # feature alignment
        self.up_2x2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up_4x4 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)

        self.fuse=CBR(in_channel*3,in_channel, kernel_size=3, padding=1)
        self.c1 = CBR(in_channel, in_channel, kernel_size=3, padding=1)
        self.c2 = CBR(in_channel, in_channel//2, kernel_size=1, padding=0)
        self.c3 = nn.Conv2d(in_channel//2, out_channel, kernel_size=1, padding=0)
        self.sig=nn.Sigmoid()

    def forward(self, x1, x2, x3):
        # feature alignment
        x2 = self.up_2x2(x2)
        x3 = self.up_4x4(x3)
        x = torch.cat([x1, x2, x3], axis=1)
        x=self.fuse(x)
        x=self.up_2x2(x)
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x=self.sig(x)
        return x


class ContrastDrivenFeatureAlignmentUnit(nn.Module):
    def __init__(self,in_channels=[64,256,512,1024],scale=[1,2,4,8],window_size=3, window_padding=1, window_stride=1):
        super().__init__()
        '''
        in_channels: channel dimension of every level feature map in model
        scale: upsample scale
        '''
        """ Adjust the shape of decouple output """
        self.preprocess_fg4 = CFAUPreprocess(in_channels[3], in_channels[3], scale[0])  # 1/16
        self.preprocess_bg4 = CFAUPreprocess(in_channels[3], in_channels[3], scale[0])  # 1/16

        self.preprocess_fg3 = CFAUPreprocess(in_channels[3], in_channels[2], scale[1])  # 1/8
        self.preprocess_bg3 = CFAUPreprocess(in_channels[3], in_channels[2], scale[1])  # 1/8

        self.preprocess_fg2 = CFAUPreprocess(in_channels[3], in_channels[1],scale[2])  # 1/4
        self.preprocess_bg2 = CFAUPreprocess(in_channels[3], in_channels[1], scale[2])  # 1/4

        self.preprocess_fg1 = CFAUPreprocess(in_channels[3], in_channels[0],scale[3])  # 1/2
        self.preprocess_bg1 = CFAUPreprocess(in_channels[3], in_channels[0], scale[3])  # 1/2


        """ Contrast-Driven Feature Aggregation """
        self.up2X = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.cfau4 = ContrastDrivenFeatureAlignment(
            in_channels[3], in_channels[3], scale[2],window_size=window_size, window_padding=window_padding, window_stride=window_stride)
        self.cfau3 = ContrastDrivenFeatureAlignment(
            in_channels[3] + in_channels[2], in_channels[2], scale[2],window_size=window_size, window_padding=window_padding, window_stride=window_stride)
        self.cfau2 = ContrastDrivenFeatureAlignment(
            in_channels[2] + in_channels[1], in_channels[1], scale[2],window_size=window_size, window_padding=window_padding, window_stride=window_stride)
        self.cfau1 = ContrastDrivenFeatureAlignment(
            in_channels[1] + in_channels[0], in_channels[0], scale[2],window_size=window_size, window_padding=window_padding, window_stride=window_stride)


    def forward(self, f_fg,f_bg,x1,x2,x3,x4):


        """ Contrast-Driven Feature Aggregation """
        f_fg4 = self.preprocess_fg4(f_fg)
        f_bg4 = self.preprocess_bg4(f_bg)
        f_fg3 = self.preprocess_fg3(f_fg)
        f_bg3 = self.preprocess_bg3(f_bg)
        f_fg2 = self.preprocess_fg2(f_fg)
        f_bg2 = self.preprocess_bg2(f_bg)
        f_fg1 = self.preprocess_fg1(f_fg)
        f_bg1 = self.preprocess_bg1(f_bg)

        """ Contrast-Driven Feature Aggregation """
        f4 = self.cfau4(x4, f_fg4, f_bg4)
        f4_up = self.up2X(f4)
        f_4_3 = torch.cat([x3, f4_up], dim=1)
        f3 = self.cfau3(f_4_3, f_fg3, f_bg3)
        f3_up = self.up2X(f3)
        f_3_2 = torch.cat([x2, f3_up], dim=1)
        f2 = self.cfau2(f_3_2, f_fg2, f_bg2)
        f2_up = self.up2X(f2)
        f_2_1 = torch.cat([x1, f2_up], dim=1)
        f1 = self.cfau1(f_2_1, f_fg1, f_bg1)

        return f1, f2, f3, f4

class CFAUPreprocess(nn.Module):

    def __init__(self, in_channel, out_channel, up_scale):
        super().__init__()
        '''
        Alignment preprocess
        '''
        up_times = int(math.log2(up_scale))
        self.preprocess = nn.Sequential()
        self.c1 = CBR(in_channel, out_channel, kernel_size=3, padding=1)
        for i in range(up_times):
            self.preprocess.add_module(f'up_{i}', nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True))
            self.preprocess.add_module(f'conv_{i}', CBR(out_channel, out_channel, kernel_size=3, padding=1))

    def forward(self, x):
        x = self.c1(x)
        x = self.preprocess(x)
        return x


class FeatureDecouplingUnit(nn.Module):
    def __init__(self, in_channel=1024,out_channel=1024):
        super().__init__()
        """
        Args:
            in_channels : output channels of last layer of model. 
            out_channel : output channels of last layer of model. 
        """
        """ Decouple Layer """
        self.fdu = FeatureDecoupling(in_channel, out_channel)
        """ Prediction Head """
        self.fdu_ph = FeatureDecouplingPredictionHead(out_channel)


    def forward(self, x):
        '''
        x : (B,C,H,W)
        '''
        """ Decouple Layer """
        f_fg, f_bg, f_uc = self.fdu(x)
        """ Prediction Head """
        mask_fg, mask_bg, mask_uc = self.fdu_ph(f_fg, f_bg, f_uc)
        
        return f_fg, f_bg, f_uc,mask_fg, mask_bg, mask_uc


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        """ Backbone: ResNet50 """
        backbone = resnet50()
        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)  # [batch_size, 64, h/2, w/2]
        self.layer1 = nn.Sequential(backbone.maxpool, backbone.layer1)  # [batch_size, 256, h/4, w/4]
        self.layer2 = backbone.layer2  # [batch_size, 512, h/8, w/8]
        self.layer3 = backbone.layer3  # [batch_size, 1024, h/16, w/16]

    def forward(self, image):
        """ Backbone: ResNet50 """
        x0 = image
        x1 = self.layer0(x0)  ## [-1, 64, h/2, w/2]
        x2 = self.layer1(x1)  ## [-1, 256, h/4, w/4]
        x3 = self.layer2(x2)  ## [-1, 512, h/8, w/8]
        x4 = self.layer3(x3)  ## [-1, 1024, h/16, w/16]

        return x1, x2, x3, x4


class Backbone_PVT(nn.Module):
    def __init__(self):
        super().__init__()
        """ Backbone: ResNet50 """
        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = 'pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

    def forward(self, image):
        """ Backbone: PVT """
        pvt = self.backbone(image)
        x1 = pvt[0] # [B, 64, h/4, w/4]
        x2 = pvt[1] # [B, 128, h/8, w/8]
        x3 = pvt[2] # [B, 320, h/16, w/16]
        x4 = pvt[3] # [B, 512, h/32, w/32]
        return x1, x2, x3, x4


class Net(nn.Module):
    def __init__(self,in_channels=[64, 256, 512, 1024],scale=[1,2,4,8],final_channel=128,window_size=3, window_padding=1, window_stride=1):
        super().__init__()
        """
        Args:
            in_channels (List[int]): Input channels for each level of model. 
            scale (List[int]): Upsampleing feature maps and configuring the number of multi-head local window attetion.
            final_channel (int): Normalizing final output channels. 
            window_size (int): Local window size for CFAU. 
        """

        """ Backbone: ResNet50 """
        self.backbone = Backbone()

        """ Decouple Layer """
        self.fdu = FeatureDecouplingUnit(in_channel=in_channels[3], out_channel=in_channels[3])
        
        """ CDFA is used to integrate with Background, foreground and encoding feature maps. """
        self.cfau=ContrastDrivenFeatureAlignmentUnit(in_channels=in_channels,scale=scale,window_size=window_size,window_padding=window_padding,window_stride=window_stride)
        
        """ Cross-level and multi-scale fature interaction"""
        self.cfcu=CrossLevelFeatureCascadeUnit(in_channel=in_channels,final_channel=final_channel,scale=2)

        """ Generating final masks """
        self.FPH = FinalPredictionHead(128, 1)

    def forward(self, image):
        '''
        Input original images: (B,3,H,W)
        Output prediction masks: (B,1,H,W)
        '''
        
        """ Backbone: ResNet50 """
        x1,x2,x3,x4=self.backbone(image)

        """ Decoupling foreground and background feature maps and generating  """
        f_fg, f_bg, _, mask_fg, mask_bg, mask_uc = self.fdu(x4)
        
        """ Contrast-Driven Feature Aggregation """
        f1, f2, f3, f4 = self.cfau(f_fg,f_bg,x1,x2,x3,x4)

        """ cross-level feature cascade and multi-scale deforming feature fusion  """
        f_small, f_middle, f_large = self.cfcu(f1, f2, f3, f4)

        """ Generating final masks """
        mask = self.FPH(f_small, f_middle, f_large)

        return mask, mask_fg, mask_bg, mask_uc



if __name__ == "__main__":
    #initialize the model
    model = Net().cuda()
    #generate random input tensor
    input_tensor = torch.randn(1, 3, 256, 256).cuda()
    #reference the model
    output = model(input_tensor)
    print(output.shape)