import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile

from lib.module import FeatureRefineBlock, NeighborConnectionDecoder, Octave, FrequencyEnhancedModule, compute_attention
from lib.PVTv2 import pvt_v2_b4


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
    def __init__(self, channels=128):
        super(FMFS, self).__init__()
        # 实例化 backbone
        self.shared_encoder = pvt_v2_b4()

        # 加载预训练权重
        state_dict = torch.load('../FMFS-Net/snapshot/pvt_v2_b4.pth')

        # 加载权重并忽略不匹配的 head 层
        missing_keys, unexpected_keys = self.shared_encoder.load_state_dict(state_dict, strict=False)

        print("PVTv2 Missing keys:", missing_keys)
        print("PVTv2 Unexpected keys:", unexpected_keys)



        self.conv_4 = ConvBlock(64)
        self.conv_3 = ConvBlock(160)
        self.conv_2 = ConvBlock(256)
        self.conv_1 = ConvBlock(128)

        # (8, 1, 28, 28) == > (8, 128, 7, 7)
        self.Up_1 = nn.Sequential(
            nn.PixelUnshuffle(4),
            nn.Conv2d(in_channels=16, out_channels=channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        self.octave_1 = Octave(in_channels=128, out_channels=channels, kernel_size=(3, 3))
        self.octave_2 = Octave(in_channels=320, out_channels=channels, kernel_size=(3, 3))
        self.octave_3 = Octave(in_channels=512, out_channels=channels, kernel_size=(3, 3))
        self.ncd = NeighborConnectionDecoder(channels)

        self.frb_4 = FeatureRefineBlock(in_channel=640, out_channel=512)
        self.frb_3 = FeatureRefineBlock(in_channel=448, out_channel=320)
        self.frb_2 = FeatureRefineBlock(in_channel=208, out_channel=128)
        self.frb_1 = FeatureRefineBlock(in_channel=96, out_channel=64)

        self.fem_4 = FrequencyEnhancedModule(32 ,2)
        self.fem_3 = FrequencyEnhancedModule(80, 2)
        self.fem_2 = FrequencyEnhancedModule(128, 3)
        self.fem_1 = FrequencyEnhancedModule(64, 4)



    def forward(self, x):
        stages = self.shared_encoder(x)

        '''
        s1 shape: torch.Size([8, 64, 56, 56])
        s2 shape: torch.Size([8, 128, 28, 28])
        s3 shape: torch.Size([8, 320, 14, 14])
        s4 shape: torch.Size([8, 512, 7, 7])
        '''
        s1, s2, s3, s4 = stages

        o1 = self.octave_1(s2)
        o2 = self.octave_2(s3)
        o3 = self.octave_3(s4)

        output_ncd = self.ncd(o3, o2, o1)                                                        # (8, 1, 28, 28)
        output_en = output_ncd.expand(-1, 32, -1, -1)                                            # (8, 32, 28, 28)
        n1 = self.Up_1(output_ncd)                                                               # (8, 128, 7, 7)


        # 4th stage
        x4 = torch.cat((s4, n1), 1)
        x4 = self.frb_4(x4)                                                                      # (8, 640, 7, 7)
        x4_up_1 = F.pixel_shuffle(x4, 2)                                            # (8, 128, 14, 14)
        x4_up_2 = F.pixel_shuffle(x4_up_1, 2)                                       # (8, 32, 28, 28)
        x4_f = self.fem_4(torch.cat((x4_up_2, output_en), 1))                       # (8, 32, 28, 28)

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


        f4_en = f4.expand(-1, 32, -1, -1)                                                       # (8, 32, 28, 28)
        f4_en_m = f4_en.mul(x4_up_2)                                                            # (8, 32, 28, 28)
        x4_f_a = torch.add(self.conv_4(torch.cat((x4_f, f4_en_m),1)), output_ncd)   # (8, 1, 28, 28)
        x4_f_a_en1 = x4_f_a.expand(-1, 80, -1, -1)                                              # (8, 80, 28, 28)
        x4_f_a_en2 = x4_f_a.expand(-1, 128, -1, -1)                                             # (8, 128, 28, 28)
        x4_f_a_en3 = F.interpolate(x4_f_a, size=(56,56),
                                   mode='bilinear',align_corners=True).expand(-1, 64, -1, -1)   # (8, 64, 56, 56)


        # 3rd stage
        x3 = torch.cat((s3, x4_up_1), 1)
        x3 = self.frb_3(x3)                                                                     # (8, 448, 14, 14)
        x3_up = F.pixel_shuffle(x3, 2)                                             # (8, 80, 28, 28)
        x3_f =self.fem_3(torch.cat((x3_up, x4_f_a_en1), 1))                        # (8, 80, 28, 28)

        f3 = compute_attention(x4_f_a)                                                          # (8, 1, 28, 28)
        f3_en = f3.expand(-1, 80, -1, -1)                                                       # (8, 80, 28, 28)
        f3_en_m = f3_en.mul(x3_up)                                                              # (8, 80, 28, 28)
        x3_f_a = torch.add(self.conv_3(torch.cat((x3_f, f3_en_m),1)), x4_f_a)      # (8, 1, 28, 28)
        x3_f_a_en1 = x3_f_a.expand(-1, 128, -1, -1)                                            # (8, 128, 28, 28)
        x3_f_a_en2 = F.interpolate(x3_f_a, size=(56,56),
                                   mode='bilinear',align_corners=True).expand(-1, 64, -1, -1)  # (8, 64, 56, 56)


        # 2ed stage
        x2 = torch.cat((s2, x3_up), 1)
        x2 = self.frb_2(x2)                                                                     # (8, 208, 28, 28)
        x2_up = F.pixel_shuffle(x2, 2)                                             # (8, 32, 56, 56)
        x2_f = self.fem_2(torch.cat((x2, x3_f_a_en1, x4_f_a_en2 ), 1))             # (8, 128, 28, 28)

        f2 = compute_attention(x3_f_a) + f3                                                     # (8, 1, 28, 28)
        f2_en = f2.expand(-1, 128, -1, -1)                                                      # (8, 128, 28, 28)
        f2_en_m = f2_en.mul(x2)                                                                 # (8, 128, 28, 28)
        x2_f_a = torch.add(self.conv_2(torch.cat((x2_f, f2_en_m), 1)), x3_f_a)     # (8, 1, 28, 28)
        x2_f_a_en1 = F.interpolate(x2_f_a, size=(56,56), mode='bilinear',align_corners=True)     # (8, 1, 56, 56)
        x2_f_a_en2 = x2_f_a_en1.expand(-1, 64, -1, -1)


        # 1st stage
        x1 = torch.cat((s1, x2_up), 1)
        x1 = self.frb_1(x1)                                                                      # (8, 96, 56, 56)
        x1_f = self.fem_1(torch.cat((x1, x2_f_a_en2, x3_f_a_en2, x4_f_a_en3 ), 1))   # (8, 64, 56, 56)

        f1 = compute_attention(x2_f_a) + f2                                                      # (8, 1, 56, 56)
        f1_en = F.interpolate(f1, size=(56,56),
                                   mode='bilinear',align_corners=True).expand(-1, 64, -1, -1)     # (8, 64, 56, 56)
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







# # Calculate FLOPs and MACs(PVTv2)   FLOPs:21.60G   MACs:21.56G   parameters: 114.81 M
# from fvcore.nn import FlopCountAnalysis
# from torchprofile import profile_macs
#
#
# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)
#
# device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
#
#
# model = FMFS().to(device)
# model.eval()
#
# x = torch.randn(1, 3, 224, 224).to(device)
#
#
# with torch.no_grad():
#     out = model(x)
#     if isinstance(out, tuple):
#         print("Full output shape:", out[0].shape)
#     else:
#         print("Full output shape:", out.shape)
#
# # FLOPs
# flops = FlopCountAnalysis(model, x)
# print(f"FLOPs: {flops.total() / 1e9:.2f} G")
#
# # MACs
# macs = profile_macs(model, args=(x,))
# print(f"MACs: {macs / 1e9:.2f} G")
#
# # 参数量 (Params)
# total_params = count_parameters(model)
# print(f"Total trainable parameters: {total_params / 1e6:.2f} M")
