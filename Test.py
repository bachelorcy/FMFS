import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from PIL import Image

# from Network.Network_MambaVision import FMFS  # mambavision
# from Network.Network_PVTv2 import FMFS  # pvt
# from Network.Network_PVTv2_NFEM import FMFS  # PVTv2(abolation FEM)
# from Network.Network_PVTv2_NFRB import FMFS  # PVTv2(abolation FRB)
from Network.Network_PVTv2_NOct import FMFS  # PVTv2(abolation Octave)
# from Network.Network_Abolation import FMFS  # Abolation
from utils.metrics import Metric


class TestDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.image_dir = os.path.join(root, 'Img')
        self.label_dir = os.path.join(root, 'GT')

        self.image_files = [f for f in os.listdir(self.image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        base_name = os.path.splitext(image_name)[0]
        image_path = os.path.join(self.image_dir, image_name)
        label_path = os.path.join(self.label_dir, base_name + ".png")

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(label_path).convert("L")  # 灰度图

        if self.transform:
            image = self.transform(image)
            mask = transforms.ToTensor()(mask)  # [0,1] 范围

        return image, mask, image_name


def load_dataset(datasets, base_dir, img_size):
    dataset_loaders = {}
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    for dataset in datasets:
        dataset_path = os.path.join(base_dir, dataset)
        test_dataset = TestDataset(root=dataset_path, transform=transform)
        dataset_loaders[dataset] = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return dataset_loaders


def test_model(model, device, save_pred_dir, dataset_loaders, dataset_names):
    model.eval()
    os.makedirs(save_pred_dir, exist_ok=True)

    total_results = {}

    with torch.no_grad():
        for dataset_name in dataset_names:
            dataloader = dataset_loaders[dataset_name]
            total_images = len(dataloader.dataset)
            pbar = tqdm(total=total_images, desc=f"Processing dataset: {dataset_name}", unit="img")

            save_dir = os.path.join(save_pred_dir, dataset_name)
            os.makedirs(save_dir, exist_ok=True)

            metric = Metric()

            for images, masks, image_names in dataloader:
                images = images.to(device, dtype=torch.float32)
                masks = masks.to(device, dtype=torch.float32)

                outputs = model(images)
                # 根据 masks 的实际尺寸调整 outputs 尺寸
                res = F.interpolate(outputs, size=(masks.shape[2], masks.shape[3]), mode='bilinear', align_corners=True)
                res = res.sigmoid().cpu().numpy()  # [B, H, W]
                masks_np = (masks.cpu().numpy() > 0.5).astype(np.uint8)  # [B, H, W]

                for i, (pred_i, mask_i, image_name) in enumerate(zip(res, masks_np, image_names)):
                    pred_uint8 = (pred_i.squeeze() * 255).astype(np.uint8)  # [H, W]
                    mask_uint8 = (mask_i.squeeze() * 255).astype(np.uint8)  # [H, W]

                    # 确保维度和形状一致
                    assert pred_uint8.ndim == 2 and mask_uint8.ndim == 2, \
                        f"维度不匹配：pred={pred_uint8.shape}, gt={mask_uint8.shape}"
                    assert pred_uint8.shape == mask_uint8.shape, \
                        f"形状不一致：pred={pred_uint8.shape}, gt={mask_uint8.shape}"

                    pred_name = os.path.splitext(os.path.basename(image_name))[0] + ".png"
                    save_img_path = os.path.join(save_dir, pred_name)
                    cv2.imwrite(save_img_path, pred_uint8)

                    # 更新指标
                    metric.step(pred=pred_uint8, gt=mask_uint8)

                pbar.update(1)

            pbar.close()
            print(f"Finished inference on dataset: {dataset_name}")

            results = metric.get_results(bit_width=4)
            total_results[dataset_name] = results

            log_str = f"[{dataset_name}] Evaluation Results: | "
            log_str += " | ".join([f"{k}: {v}" for k, v in results.items()])
            print(log_str)

        print("Prediction and evaluation completed for all datasets.")
        return total_results


def main(config):
    test_loader = load_dataset(config['datasets'], config['base_dir'], config['img_size'])

    # 初始化模型
    model = FMFS()

    checkpoint = torch.load(config['checkpoint_path'])

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint  # 假设是直接的 state_dict

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print(f"FMFS Missing keys: {missing_keys}")
    print(f"FMFS Unexpected keys: {unexpected_keys}")

    model.to(config['device'])
    model.eval()

    # 执行测试并保存预测结果
    results = test_model(
        model=model,
        device=config['device'],
        save_pred_dir=config['save_dir'],
        dataset_loaders=test_loader,
        dataset_names=config['datasets']
    )

    return results


if __name__ == "__main__":
    config = {
        'datasets': [ 'CAMO', 'COD10K', 'NC4K'],
        'base_dir': '../FMFS-Net/data/Test/',
        'save_dir': '../FMFS-Net/Results',
        'img_size': 224,
        'checkpoint_path': '../FMFS-Net/checkpoints/Net_epoch_best(oct).pth',
        'device': 'cuda:2'
    }

    main(config)