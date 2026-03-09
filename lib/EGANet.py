import torch
import torch.nn.functional as F
import torch.nn as nn
from .modules import *
import torchvision.models as models
from torchvision.transforms.functional import rgb_to_grayscale
from .res2net import res2net50_v1b_26w_4s
import sys
import os

# 将根目录下的 dino 文件夹加入环境变量，以本地加载 DINO 的 vision_transformer
dino_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../dino'))
if dino_path not in sys.path:
    sys.path.append(dino_path)
import vision_transformer as vits  # type:ignore


class Conv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class Up(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            Conv(in_channels, in_channels // 4), 
            Conv(in_channels // 4, out_channels)

        )
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x1, x2):
        x = torch.cat([x2, x1], dim=1)
        out = self.conv(x)
        return self.up(out)


class Out(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Out, self).__init__()
        self.conv1 = Conv(in_channels, in_channels // 4, kernel_size=kernel_size,
                               stride=stride, padding=padding)

        self.conv2 = nn.Conv2d(in_channels // 4, out_channels, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


# Multi-Scale Edge-Guided Attention Network with DINO Fusion
class EGANet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(EGANet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # DION SetUp
        print("Initializing Local DINO Backbone (ViT-Small, Patch=8)...")
        self.dino = vits.__dict__['vit_small'](patch_size=8, num_classes=0)

        # 本地加载预训练权重
        dino_weight_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '../../dino_deitsmall8_pretrain.pth'))
        if os.path.exists(dino_weight_path):
            dino_state = torch.load(dino_weight_path, map_location='cpu')
            self.dino.load_state_dict(dino_state, strict=False)
            print(f"Successfully loaded DINO weights from {dino_weight_path}")
        else:
            print(f"Warning: DINO weight not found at {dino_weight_path}. Using random weights.")

        # 冻结DINO所有的参数
        for p in self.dino.parameters():
            p.requires_grad = False
        self.dino.eval()

        # Res2Net Encoder
        resnet = res2net50_v1b_26w_4s(pretrained=True)
        self.encoder1_conv = resnet.conv1
        self.encoder1_bn = resnet.bn1
        self.encoder1_relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.encoder2 = resnet.layer1
        self.encoder3 = resnet.layer2
        self.encoder4 = resnet.layer3
        self.encoder5 = resnet.layer4

        # DINO-Res2Net Fusion Layers
        dino_dim = 384
        self.dino_fuse3 = nn.Sequential(nn.Conv2d(512 + dino_dim, 512, 3, padding=1), nn.BatchNorm2d(512),
                                        nn.ReLU(inplace=True))
        self.dino_fuse4 = nn.Sequential(nn.Conv2d(1024 + dino_dim, 1024, 3, padding=1), nn.BatchNorm2d(1024),
                                        nn.ReLU(inplace=True))
        self.dino_fuse5 = nn.Sequential(nn.Conv2d(2048 + dino_dim, 2048, 3, padding=1), nn.BatchNorm2d(2048),
                                        nn.ReLU(inplace=True))

        # Dimensionality Reduction
        self.x5_dem_1 = nn.Sequential(nn.Conv2d(2048, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True))
        self.x4_dem_1 = nn.Sequential(nn.Conv2d(1024, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.x3_dem_1 = nn.Sequential(nn.Conv2d(512, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.x2_dem_1 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        # Decoder
        self.up5 = nn.Sequential(
            Conv(512, 512),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        ) 
        self.up4 = Up(768, 256)
        self.up3 = Up(384, 128)
        self.up2 = Up(192, 64)
        self.up1 = Up(128, 64)

        self.ega1 = EGA(64)
        self.ega2 = EGA(64)
        self.ega3 = EGA(128)
        self.ega4 = EGA(256)
        
        self.out5 = Out(512, n_classes)
        self.out4 = Out(256, n_classes)
        self.out3 = Out(128, n_classes)
        self.out2 = Out(64, n_classes)
        self.out1 = Out(64, n_classes)

    def train(self, mode=True):
        """重写train方法，强制使得dino保持在eval模式，避免BN或Dropout更新"""
        super(EGANet, self).train(mode)
        if hasattr(self, 'dino'):
            self.dino.eval()

    def forward(self, x):
        grayscale_img = rgb_to_grayscale(x)
        edge_feature = make_laplace_pyramid(grayscale_img, 5, 1)
        edge_feature = edge_feature[0]

        # 提取DINO特征图 (No Gradients)
        with torch.no_grad():
            self.dino.eval()  # 永远处于eval模式
            B, C, H, W = x.shape
            w_feat, h_feat = W // 8, H // 8
            # 获取DINO特征 [B, N, 384]，N = 1(CLS) + w_feat * h_feat
            dino_out = self.dino.get_intermediate_layers(x, n=1)[0]
            dino_feat = dino_out[:, 1:]  # 移除CLS Token
            # 将Token序列Reshape回空间2D特征图 [B, 384, H/8, W/8]
            dino_feat = dino_feat.transpose(1, 2).reshape(B, 384, h_feat, w_feat)

        # Encoder(Res2Net)
        e1 = self.encoder1_conv(x)  
        e1 = self.encoder1_bn(e1)
        e1 = self.encoder1_relu(e1)
        e1_pool = self.maxpool(e1)  
        
        e2 = self.encoder2(e1_pool)

        e3 = self.encoder3(e2)
        # Fusion 1: e3 (1/8 resolution) 融合 DINO 特征 (1/8 resolution)
        e3_fused = torch.cat([e3, dino_feat], dim=1)
        e3 = self.dino_fuse3(e3_fused)
        
        e4 = self.encoder4(e3)
        # Fusion 2: e4 (1/16 resolution) 融合 DINO 降采样特征
        dino_feat_pool1 = F.max_pool2d(dino_feat, kernel_size=2, stride=2)
        e4_fused = torch.cat([e4, dino_feat_pool1], dim=1)
        e4 = self.dino_fuse4(e4_fused)

        e5 = self.encoder5(e4)
        # Fusion 3: e5 (1/32 resolution) 融合 DINO 降采样特征
        dino_feat_pool2 = F.max_pool2d(dino_feat_pool1, kernel_size=2, stride=2)
        e5_fused = torch.cat([e5, dino_feat_pool2], dim=1)
        e5 = self.dino_fuse5(e5_fused)

        # 下采样与特征维度降低
        e5_dem_1 = self.x5_dem_1(e5)
        e4_dem_1 = self.x4_dem_1(e4)
        e3_dem_1 = self.x3_dem_1(e3)
        e2_dem_1 = self.x2_dem_1(e2)

        # Decoder
        d5 = self.up5(e5_dem_1)
        out5 = self.out5(d5)
        ega4 = self.ega4(edge_feature, e4_dem_1, out5)

        d4 = self.up4(d5, ega4)
        out4 = self.out4(d4)    
        ega3 = self.ega3(edge_feature, e3_dem_1, out4)

        d3 = self.up3(d4, ega3)
        out3 = self.out3(d3)
        ega2 = self.ega2(edge_feature, e2_dem_1, out3)

        d2 = self.up2(d3, ega2)
        out2 = self.out2(d2)  
        ega1 = self.ega1(edge_feature, e1, out2)

        d1 = self.up1(d2, ega1)
        out1 = self.out1(d1)
        
        return out1, out2, out3, out4, out5


class EGANetModel(nn.Module): 
    def __init__(self, n_channels=3, n_classes=1):
        super(EGANetModel, self).__init__()
        self.channel = n_channels
        self.num_classes = n_classes
        self.net = EGANet(self.channel, self.num_classes)

    def forward(self, images):
        out1, out2, out3, out4, out5 = self.net(images)
        return out1, out2, out3, out4, out5
