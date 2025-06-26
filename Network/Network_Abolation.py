import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.PVTv2 import pvt_v2_b4


class FMFS(nn.Module):
    def __init__(self):
        super(FMFS, self).__init__()
        # 实例化 backbone
        self.shared_encoder = pvt_v2_b4()

        # 加载预训练权重
        state_dict = torch.load('../FMFS-Net/snapshot/pvt_v2_b4.pth')

        # 加载权重并忽略不匹配的 head 层
        missing_keys, unexpected_keys = self.shared_encoder.load_state_dict(state_dict, strict=False)

        print("PVTv2 Missing keys:", missing_keys)
        print("PVTv2 Unexpected keys:", unexpected_keys)

        self.Up_4 = nn.Sequential(
            nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False),
            nn.Conv2d(512, 1, kernel_size=1)
        )


    def forward(self, x):
        stages = self.shared_encoder(x)

        '''
        s1 shape: torch.Size([8, 64, 56, 56])
        s2 shape: torch.Size([8, 128, 28, 28])
        s3 shape: torch.Size([8, 320, 14, 14])
        s4 shape: torch.Size([8, 512, 7, 7])
        '''
        s1, s2, s3, s4 = stages

        s4_en = self.Up_4(s4)


        return s4_en


# # Calculate FLOPs and MACs(PVTv2)   FLOPs:10.27 G   MACs:10.24 G   parameters: 62.04 M
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