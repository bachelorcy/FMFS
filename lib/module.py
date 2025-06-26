import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_dct as dct
from pytorch_wavelets import DWTForward, DWTInverse

'''
Octave convolution

@article{chen2019drop,
  title={Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural Networks with Octave Convolution},
  author={Chen, Yunpeng and Fan, Haoqi and Xu, Bing and Yan, Zhicheng and Kalantidis, Yannis and Rohrbach, Marcus and Yan, Shuicheng and Feng, Jiashi},
  journal={Proceedings of the IEEE International Conference on Computer Vision},
  year={2019}
}
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class FirstOctaveConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False):
        super(FirstOctaveConv, self).__init__()
        self.stride = stride
        kernel_size = kernel_size[0]


        self.h2g_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)

        self.h2l = torch.nn.Conv2d(in_channels, int(alpha * in_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)

        self.h2h = torch.nn.Conv2d(in_channels, in_channels - int(alpha * in_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)

        self.bn_h = nn.GroupNorm(num_groups=1, num_channels=self.h2h.out_channels)
        self.bn_l = nn.GroupNorm(num_groups=1, num_channels=self.h2l.out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.stride == 2:
            x = self.h2g_pool(x)

        X_h2l = self.h2g_pool(x)

        X_h = self.h2h(x)
        X_l = self.h2l(X_h2l)

        X_h = self.bn_h(X_h)
        X_l = self.bn_l(X_l)
        X_h = self.relu(X_h)
        X_l = self.relu(X_l)

        return X_h, X_l


class OctaveConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False):
        super(OctaveConv, self).__init__()
        kernel_size = kernel_size[0]

        self.h2g_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')
        self.stride = stride

        self.l2l = torch.nn.Conv2d(int(alpha * in_channels), int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.l2h = torch.nn.Conv2d(int(alpha * in_channels), out_channels - int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.h2l = torch.nn.Conv2d(in_channels - int(alpha * in_channels), int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.h2h = torch.nn.Conv2d(in_channels - int(alpha * in_channels),
                                   out_channels - int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)

        high_ch = out_channels - int(alpha * out_channels)
        low_ch = int(alpha * out_channels)
        self.gate_conv_h = nn.Conv2d(high_ch * 2, high_ch, kernel_size=1)
        self.gate_conv_l = nn.Conv2d(low_ch * 2, low_ch, kernel_size=1)

        self.bn_h = nn.GroupNorm(num_groups=1, num_channels=high_ch)
        self.bn_l = nn.GroupNorm(num_groups=1, num_channels=low_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        X_h, X_l = x

        if self.stride == 2:
            X_h, X_l = self.h2g_pool(X_h), self.h2g_pool(X_l)

        X_h2l = self.h2g_pool(X_h)
        X_h2h = self.h2h(X_h)
        X_l2h = self.l2h(X_l)

        X_l2l = self.l2l(X_l)
        X_h2l = self.h2l(X_h2l)


        X_l2h = F.interpolate(X_l2h, (X_h2h.size(2), X_h2h.size(3)), mode='bilinear')
        gate_h = torch.cat([X_l2h, X_h2h], dim=1)
        gate_h = self.gate_conv_h(gate_h)
        gate_h = torch.sigmoid(gate_h)
        X_h = gate_h * X_l2h + (1 - gate_h) * X_h2h

        gate_l = torch.cat([X_h2l, X_l2l], dim=1)
        gate_l = self.gate_conv_l(gate_l)
        gate_l = torch.sigmoid(gate_l)
        X_l = gate_l * X_h2l + (1 - gate_l) * X_l2l

        X_h = self.bn_h(X_h)
        X_l = self.bn_l(X_l)
        X_h = self.relu(X_h)
        X_l = self.relu(X_l)

        return X_h, X_l


class LastOctaveConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False):
        super(LastOctaveConv, self).__init__()
        self.stride = stride
        kernel_size = kernel_size[0]


        self.h2g_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.l2h = torch.nn.Conv2d(int(alpha * out_channels), out_channels,
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.h2h = torch.nn.Conv2d(out_channels - int(alpha * out_channels),
                                   out_channels,
                                   kernel_size, 1, padding, dilation, groups, bias)

        self.upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')
        self.bn = nn.GroupNorm(num_groups=1, num_channels=out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        X_h, X_l = x

        if self.stride == 2:
            X_h, X_l = self.h2g_pool(X_h), self.h2g_pool(X_l)

        X_h2h = self.h2h(X_h)
        X_l2h = self.l2h(X_l)

        X_l2h = F.interpolate(X_l2h, (int(X_h2h.size()[2]), int(X_h2h.size()[3])), mode='bilinear')

        X_h = X_h2h + X_l2h
        X_h = self.bn(X_h)
        X_h = self.relu(X_h)

        return X_h       # Return the final single output


class Octave(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3)):
        super(Octave, self).__init__()

        self.fir = FirstOctaveConv(in_channels, out_channels, kernel_size)

        self.mid1 = OctaveConv(in_channels, in_channels, kernel_size)
        self.mid2 = OctaveConv(in_channels, out_channels, kernel_size)

        self.lst = LastOctaveConv(in_channels, out_channels, kernel_size)

    def forward(self, x):
        x0 = x

        x_h, x_l = self.fir(x)
        x_hh, x_ll = x_h, x_l,

        x_h_1, x_l_1 = self.mid1((x_h, x_l))
        x_h_2, x_l_2 = self.mid1((x_h_1, x_l_1))

        x_h_5, x_l_5 = self.mid2((x_h_2, x_l_2))
        x_ret = self.lst((x_h_5, x_l_5))

        return x_ret



'''
NCD

@inproceedings{fan2020camouflaged,
  title={Camouflaged object detection},
  author={Fan, Deng-Ping and Ji, Ge-Peng and Sun, Guolei and Cheng, Ming-Ming and Shen, Jianbing and Shao, Ling},
  booktitle={IEEE CVPR},
  pages={2777--2787},
  year={2020}
}
'''

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class NeighborConnectionDecoder(nn.Module):
    def __init__(self, channel):
        super(NeighborConnectionDecoder, self).__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)


        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)


        self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)

        self.conv4 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3 * channel, 1, 1)

    def forward(self, x1, x2, x3):  # Deep features -> shallow features


        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2

        x3_1 = self.conv_upsample2(self.upsample(x2_1)) * self.conv_upsample3(self.upsample(x2)) * x3
        x3_1 = F.relu(x3_1)

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        x = self.conv5(x)

        return x


'''
Feature Refine Block(FRB)
'''

class EnhancedReceptionBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(EnhancedReceptionBlock, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        self.Conv0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )

        self.Conv1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )

        self.Conv2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )

        self.Conv3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )

        self.conv_cat = BasicConv2d(4 * out_channel, out_channel, 3, padding=1)

    def forward(self, x):
        x0 = self.Conv0(x)
        x1 = self.Conv1(x)
        x2 = self.Conv2(x)
        x3 = self.Conv3(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + x0)
        return x




class SpatialAttention(nn.Module):  # 空间注意力模块
    def __init__(self, in_channel):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        MaxPool = torch.max(x, dim=1).values
        AvgPool = torch.mean(x, dim=1)

        MaxPool = torch.unsqueeze(MaxPool, dim=1)
        AvgPool = torch.unsqueeze(AvgPool, dim=1)

        x_cat = torch.cat((MaxPool, AvgPool), dim=1)
        x_out = self.conv(x_cat)
        Ms = self.sigmoid(x_out)

        x = Ms * x + x
        return x


class FeatureRefineBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(FeatureRefineBlock, self).__init__()
        self.enhanced_reception = EnhancedReceptionBlock(in_channel, out_channel)
        self.spatial_attention = SpatialAttention(out_channel)

        self.bn_final = nn.BatchNorm2d(out_channel)
        self.relu_final = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.enhanced_reception(x)
        x = self.spatial_attention(x)

        x = self.bn_final(x)
        x = self.relu_final(x)

        return x


'''
Frequency Enhanced Module
'''
class FrequencyEnhancedModule(nn.Module):
    def __init__(self, in_channels, s):
        super(FrequencyEnhancedModule, self).__init__()
        """
                Frequency Enhanced Feature Extraction Module
                Combines temporal feature extraction, spatial refinement, and frequency domain enhancement

                Args:
                    in_channels (int): Number of input channels
                    scale_factor (int): Channel scaling factor for initial projection (default: 2)
                """

        self.temporal_branch = nn.Sequential(
            nn.Conv2d(in_channels * s, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True)
        )

        self.spatial_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
        )

        inter_channels = max(in_channels // 16, 1)
        self.freq_attention = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=True),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(True),
            nn.Conv2d(inter_channels, in_channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        temporal_feat = self.temporal_branch(x)

        spatial_feat = self.spatial_branch(temporal_feat)

        freq_features = torch.fft.fft2(temporal_feat.float())
        freq_r = freq_features.real
        freq_w = self.freq_attention(freq_r)

        freq_en = torch.fft.ifft2(freq_w * freq_features)
        freq_en = torch.abs(freq_en)
        freq_en = self.relu(self.norm(freq_en))

        x = torch.add(spatial_feat, freq_en)

        return x



def compute_attention(x):
    x_freq = torch.fft.fft2(x)
    x_f_a = torch.abs(x_freq)
    x_f_a = -1 * (torch.sigmoid(x_f_a)) + 1

    x_s = -1 * (torch.sigmoid(x)) + 1
    x_s_f = x_s + x_f_a

    return x_s_f



