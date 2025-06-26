import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile

from lib.module import FeatureRefineBlock, NeighborConnectionDecoder, Octave, FrequencyEnhancedModule, compute_attention
from lib.mamba_vision import mamba_vision_S

class ConvBlock(nn.Module):
    def __init__(self, in_channels):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(True),
            nn.Conv2d(in_channels // 2, 1, kernel_size=1)
        )

    def forward(self, x):
        return self.block(x)


class FMFS(nn.Module):
    def __init__(self, channels=192):
        super(FMFS, self).__init__()
        # MambaVision
        self.shared_encoder = mamba_vision_S(pretrained=True, model_path='../FMFS-Net/snapshot/mambavision_small_1k.pth.tar')

        self.conv_4 = ConvBlock(96)
        self.conv_3 = ConvBlock(192)
        self.conv_2 = ConvBlock(384)
        self.conv_1 = ConvBlock(192)

        # (8, 1, 28, 28) == > (8, 192, 7, 7)
        self.Up_1 = nn.Sequential(
            nn.PixelUnshuffle(4),
            nn.Conv2d(in_channels=16, out_channels=channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        self.octave_1 = Octave(in_channels=192, out_channels=channels, kernel_size=(3, 3))
        self.octave_2 = Octave(in_channels=384, out_channels=channels, kernel_size=(3, 3))
        self.octave_3 = Octave(in_channels=768, out_channels=channels, kernel_size=(3, 3))
        self.ncd = NeighborConnectionDecoder(channels)

        self.frb_4 = FeatureRefineBlock(in_channel=960, out_channel=768)
        self.frb_3 = FeatureRefineBlock(in_channel=576, out_channel=384)
        self.frb_2 = FeatureRefineBlock(in_channel=288, out_channel=192)
        self.frb_1 = FeatureRefineBlock(in_channel=144, out_channel=96)

        self.fem_4 = FrequencyEnhancedModule(48 ,2)
        self.fem_3 = FrequencyEnhancedModule(96, 2)
        self.fem_2 = FrequencyEnhancedModule(192, 3)
        self.fem_1 = FrequencyEnhancedModule(96, 4)



    def forward(self, x):
        stages = self.shared_encoder.forward_features(x)

        '''
        s0 shape: torch.Size([B, 96, 56, 56])
        s1 shape: torch.Size([B, 192, 28, 28])
        s2 shape: torch.Size([B, 384, 14, 14])
        s3 shape: torch.Size([B, 768, 7, 7])
        s4 shape: torch.Size([B, 768, 7, 7])
        '''

        s1, s2, s3, _, s4 = stages

        o1 = self.octave_1(s2)
        o2 = self.octave_2(s3)
        o3 = self.octave_3(s4)


        output_ncd = self.ncd(o3, o2, o1)                                                        # (8, 1, 28, 28)
        output_en = output_ncd.expand(-1, 48, -1, -1)                                            # (8, 48, 28, 28)
        n1 = self.Up_1(output_ncd)                                                               # (8, 192, 7, 7)


        # 4th stage
        x4 = torch.cat((s4, n1), 1)
        x4 = self.frb_4(x4)                                                                      # (8, 768, 7, 7)
        x4_up_1 = F.pixel_shuffle(x4, 2)                                            # (8, 192, 14, 14)
        x4_up_2 = F.pixel_shuffle(x4_up_1, 2)                                       # (8, 48, 28, 28)
        x4_f = self.fem_4(torch.cat((x4_up_2, output_en), 1))                       # (8, 48, 28, 28)

        f4 = compute_attention(output_ncd)                                                      # (8, 1, 28, 28)

        # # Debug（for batchnormalization and activate）
        # print("Input:", x.min().item(), "~", x.max().item())  # debug
        # print(f"After shared_encoder s4 range: {s4.min().item():.4f} ~ {s4.max().item():.4f}")    # debug
        # print(f"After shared_encoder s3 range: {s3.min().item():.4f} ~ {s3.max().item():.4f}")  # debug
        # print(f"After shared_encoder s2 range: {s2.min().item():.4f} ~ {s2.max().item():.4f}")  # debug
        # print(f"After shared_encoder s1 range: {s1.min().item():.4f} ~ {s1.max().item():.4f}")  # debug
        # print(f"After o3 range: {o3.min().item():.4f} ~ {o3.max().item():.4f}")  # debug
        # print(f"After o2 range: {o2.min().item():.4f} ~ {o2.max().item():.4f}")  # debug
        # print(f"After o1 range: {o1.min().item():.4f} ~ {o1.max().item():.4f}")   # debug
        # print(f"After ncd range: {output_ncd.min().item():.4f} ~ {output_ncd.max().item():.4f}")  # debug
        # print(f"After Up_1: {n1.min().item():.4f} ~ {n1.max().item():.4f}")  # debug
        # print(f"After frb_4: {x4.min().item():.4f} ~ {x4.max().item():.4f}")  # debug
        # print(f"After fem_4: {x4_f.min().item():.4f} ~ {x4_f.max().item():.4f}")  # debug
        # print(f"After compute_attention: {f4.min().item():.4f} ~ {f4.max().item():.4f}")  # debug


        f4_en = f4.expand(-1, 48, -1, -1)                                                       # (8, 48, 28, 28)
        f4_en_m = f4_en.mul(x4_up_2)                                                            # (8, 48, 28, 28)
        x4_f_a = torch.add(self.conv_4(torch.cat((x4_f, f4_en_m),1)), output_ncd)   # (8, 1, 28, 28)
        x4_f_a_en1 = x4_f_a.expand(-1, 96, -1, -1)                                              # (8, 96, 28, 28)
        x4_f_a_en2 = x4_f_a.expand(-1, 192, -1, -1)                                             # (8, 192, 28, 28)
        x4_f_a_en3 = F.interpolate(x4_f_a, size=(56,56),
                                   mode='bilinear',align_corners=True).expand(-1, 96, -1, -1)   # (8, 96, 56, 56)


        # 3rd stage
        x3 = torch.cat((s3, x4_up_1), 1)
        x3 = self.frb_3(x3)                                                                     # (8, 384, 14, 14)
        x3_up = F.pixel_shuffle(x3, 2)                                             # (8, 96, 28, 28)
        x3_f =self.fem_3(torch.cat((x3_up, x4_f_a_en1), 1))                        # (8, 96, 28, 28)

        f3 = compute_attention(x4_f_a)                                                          # (8, 1, 28, 28)
        f3_en = f3.expand(-1, 96, -1, -1)                                                       # (8, 96, 28, 28)
        f3_en_m = f3_en.mul(x3_up)                                                              # (8, 96, 28, 28)
        x3_f_a = torch.add(self.conv_3(torch.cat((x3_f, f3_en_m),1)), x4_f_a)      # (8, 1, 28, 28)
        x3_f_a_en1 = x3_f_a.expand(-1, 192, -1, -1)                                            # (8, 192, 28, 28)
        x3_f_a_en2 = F.interpolate(x3_f_a, size=(56,56),
                                   mode='bilinear',align_corners=True).expand(-1, 96, -1, -1)  # (8, 96, 56, 56)


        # 2ed stage
        x2 = torch.cat((s2, x3_up), 1)
        x2 = self.frb_2(x2)                                                                     # (8, 192, 28, 28)
        x2_up = F.pixel_shuffle(x2, 2)                                             # (8, 48, 56, 56)
        x2_f = self.fem_2(torch.cat((x2, x3_f_a_en1, x4_f_a_en2 ), 1))             # (8, 192, 28, 28)

        f2 = compute_attention(x3_f_a) + f3                                                     # (8, 1, 28, 28)
        f2_en = f2.expand(-1, 192, -1, -1)                                                      # (8, 192, 28, 28)
        f2_en_m = f2_en.mul(x2)                                                                 # (8, 192, 28, 28)
        x2_f_a = torch.add(self.conv_2(torch.cat((x2_f, f2_en_m), 1)), x3_f_a)     # (8, 1, 28, 28)
        x2_f_a_en1 = F.interpolate(x2_f_a, size=(56,56), mode='bilinear',align_corners=True)     # (8, 1, 56, 56)
        x2_f_a_en2 = x2_f_a_en1.expand(-1, 96, -1, -1)


        # 1st stage
        x1 = torch.cat((s1, x2_up), 1)                                               # (8, 144, 56, 56)
        x1 = self.frb_1(x1)                                                                      # (8, 96, 56, 56)
        x1_f = self.fem_1(torch.cat((x1, x2_f_a_en2, x3_f_a_en2, x4_f_a_en3 ), 1))   # (8, 96, 56, 56)

        f1 = compute_attention(x2_f_a) + f2                                                      # (8, 1, 56, 56)
        f1_en = F.interpolate(f1, size=(56,56),
                                   mode='bilinear',align_corners=True).expand(-1, 96, -1, -1)     # (8, 96, 56, 56)
        f1_en_m = f1_en.mul(x1)
        x1_f_a = torch.add(self.conv_1(torch.cat((x1_f, f1_en_m), 1)), x2_f_a_en1)       # (8, 1, 56, 56)


        p1 = F.interpolate(x1_f_a, size=(224,224), mode='bilinear',align_corners=True)
        p2 = F.interpolate(x2_f_a, size=(224,224), mode='bilinear',align_corners=True)
        p3 = F.interpolate(x3_f_a, size=(224, 224), mode='bilinear', align_corners=True)
        p4 = F.interpolate(x4_f_a, size=(224, 224), mode='bilinear', align_corners=True)
        p5 = F.interpolate(output_ncd, size=(224,224), mode='bilinear', align_corners=False)



        if self.training:
            return p1, p2, p3, p4, p5
        else:
            return p1






# # Calculate FLOPs and MACs(mambavision)   FLOPs:31.16G   MACs:31.14G
# from fvcore.nn import FlopCountAnalysis
# from torchprofile import profile_macs
#
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = FGMVNet().to(device)
# model.eval()
#
# x = torch.randn(1, 3, 224, 224).to(device)
#
# # 测试前向传播是否正常
# with torch.no_grad():
#     out = model(x)
# print("Full output shape:", out[0].shape if isinstance(out, tuple) else out.shape)
#
# # FLOPs
# flops = FlopCountAnalysis(model, x)
# print(f"FLOPs: {flops.total() / 1e9:.2f} G")
#
# # MACs
# macs = profile_macs(model, args=(x,))
# print(f"MACs: {macs / 1e9:.2f} G")












