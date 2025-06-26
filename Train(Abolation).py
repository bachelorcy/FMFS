import os
import torch
import argparse
import logging
import cv2
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
from torch.optim.lr_scheduler import LinearLR, ReduceLROnPlateau
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from Dataset import Config, CamDataset
from utils.metrics import Metric

from Network.Network_Abolation import FMFS  # All Abolation


def parser():
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument('--batchsize', default=10, type=int, help="Batch size for training")
    parser.add_argument('--savepath', default="../FMFS-Net/checkpoints", type=str, help="Path to save checkpoints")
    parser.add_argument('--datapath', default="../FMFS-Net/data", type=str, help="Path to dataset")
    parser.add_argument('--logpath', default="../FMFS-Net/logs", type=str, help="Path to logs")
    parser.add_argument('--checkpoint', default='../FMFS-Net/checkpoints/Net_epoch_best(ABO).pth', type=str)  # All Abolation
    parser.add_argument('--val_pred', default='../FMFS-Net/val_pred', type=str, help="Path to save validation predictions")
    parser.add_argument('--lr', default=1e-5, type=float, help="Learning rate")
    parser.add_argument('--epoch', default=60, type=int, help="Number of epochs")
    parser.add_argument('--device', default="cuda:0", type=str, help="Device (cuda or cpu)")
    return parser.parse_args()


class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)

        # 初始化数据集配置
        train_cfg = Config(
            img_dir=os.path.join(args.datapath, 'Train', 'Image'),
            mask_dir=os.path.join(args.datapath, 'Train', 'GT_Object'),
            mode='train',
            resize_size=(224, 224)
        )
        val_cfg = Config(
            img_dir=os.path.join(args.datapath, 'Val', 'Image'),
            mask_dir=os.path.join(args.datapath, 'Val', 'GT_Object'),
            mode='val',
            resize_size=(224, 224)
        )

        # 初始化数据加载器
        self.train_loader = DataLoader(
            CamDataset(train_cfg),
            batch_size=args.batchsize,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        self.val_loader = DataLoader(
            CamDataset(val_cfg),
            batch_size=args.batchsize,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )

        # 初始化模型
        self.model = FMFS().to(self.device)

        # 初始化优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)

        # Warmup + Plateau 的组合调度器
        self.warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=0.1, end_factor=1.0, total_iters=5
        )
        self.main_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=3
        )
        self.use_warmup = False  # 控制是否处于 warmup 阶段

        # 创建保存路径&初始化日志
        self.logpath = args.logpath
        os.makedirs(self.logpath, exist_ok=True)
        log_file = os.path.join(self.logpath, 'training.log')
        logging.basicConfig(filename=log_file, level=logging.INFO,
                            format='%(asctime)s %(levelname)s: %(message)s')

        # 初始化 TensorBoard
        tensorboard_log_dir = os.path.join(self.logpath, 'tensorboard_logs')
        self.writer = SummaryWriter(log_dir=tensorboard_log_dir)

        # 初始化最佳指标
        self.best_score = float('inf')
        self.best_epoch = -1

        # 检查点恢复
        if args.checkpoint and os.path.exists(args.checkpoint):
            print(f"Loading checkpoint from {args.checkpoint}")
            self.model.load_state_dict(torch.load(args.checkpoint))
            self.use_warmup = False  # 禁用 warmup
            print("Checkpoint loaded successfully.")

    def train(self):
        for epoch in range(1, self.args.epoch + 1):
            start_time = datetime.now()

            if self.use_warmup and epoch <= 5:
                print(f"Warmup phase: Epoch {epoch}")
                avg_loss = self.train_epoch(epoch)
                self.warmup_scheduler.step()
            else:
                self.use_warmup = False
                avg_loss = self.train_epoch(epoch)
                val_loss, eval_mae, eval_Fm, eval_Sm, eval_Em, eval_IoU = self.validate(current_epoch=epoch)
                self.main_scheduler.step(val_loss)

                cur_score = eval_mae  # 可根据需求自定义 best score 公式
                log_message = (
                    f"Epoch {epoch}/{self.args.epoch} | "
                    f"Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f} | "
                    f"MAE: {eval_mae:.4f} | Fm: {eval_Fm:.4f} | Sm: {eval_Sm:.4f} | Em: {eval_Em:.4f} | IoU: {eval_IoU:.4f}"
                )
                print(log_message)
                logging.info(log_message)

                # 写入 TensorBoard
                self.writer.add_scalar("Loss/train", avg_loss, epoch)
                self.writer.add_scalar("Loss/val", val_loss, epoch)
                self.writer.add_scalar("Metrics/MAE", eval_mae, epoch)
                self.writer.add_scalar("Metrics/Fm", eval_Fm, epoch)
                self.writer.add_scalar("Metrics/Sm", eval_Sm, epoch)
                self.writer.add_scalar("Metrics/Em", eval_Em, epoch)
                self.writer.add_scalar("Metrics/IoU", eval_IoU, epoch)

                # 保存最佳模型
                if cur_score < self.best_score:
                    self.best_score = cur_score
                    self.best_epoch = epoch
                    save_path = os.path.join(self.args.savepath, 'Net_epoch_best(ABO).pth')
                    torch.save(self.model.state_dict(), save_path)
                    print(f">>> Save best model successfully! Best epoch is {epoch}.")

            # 每10个epoch保存一次权重
            if epoch % 10 == 0:
                save_name = f'Net_epoch_{epoch}（ABO）.pth'
                torch.save(self.model.state_dict(), os.path.join(self.args.savepath, save_name))
                print(f">>> Save state_dict successfully! Epoch is {epoch}.")

            end_time = datetime.now()
            duration = end_time - start_time
            logging.info(f"Epoch {epoch} finished. Duration: {duration}, Total Loss: {avg_loss:.4f}")

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        loop = tqdm(self.train_loader, desc=f"Epoch {epoch}", leave=True)

        for images, masks in loop:
            images = images.to(self.device, dtype=torch.float32)
            masks = masks.to(self.device, dtype=torch.float32)

            outputs = self.model(images)
            target_size = masks.shape[-2:]
            res = F.interpolate(outputs, size=target_size, mode='bilinear', align_corners=False)

            loss = self.Loss_Function(res, masks)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    def validate(self, current_epoch):
        metric = Metric()
        total_loss = 0.0
        self.model.eval()

        with torch.no_grad():
            for idx, (images, masks) in enumerate(self.val_loader):
                images = images.to(self.device, dtype=torch.float32)
                masks = masks.to(self.device, dtype=torch.float32)

                outputs = self.model(images)
                target_size = masks.shape[-2:]
                res = F.interpolate(outputs, size=target_size, mode='bilinear', align_corners=False)

                loss = self.Structure_loss(res, masks).item()
                total_loss += loss

                # 转换为 numpy，并映射到 [0, 255] 并转为 uint8
                res_np = (res.sigmoid().cpu().numpy() * 255).astype(np.uint8)
                masks_np = (masks.cpu().numpy() * 255).astype(np.uint8)

                for pred_i, gt_i in zip(res_np, masks_np):
                    metric.step(pred=pred_i.squeeze(), gt=gt_i.squeeze())

            results = metric.get_results(bit_width=4)
            avg_loss = total_loss / len(self.val_loader)

            log_str = f"Epoch {current_epoch}/100 | "
            log_str += f"Val Loss: {avg_loss:.4f} | "
            log_str += " | ".join([f"{k}: {v}" for k, v in results.items()])
            print(log_str)

            return avg_loss, float(results["MAE"]), float(results["meanFm"]), float(results["Smeasure"]), float(
                results["meanEm"]), float(results["Mean_IoU"])

    def Loss_Function(self, preds, mask):
        return self.Structure_loss(preds, mask)

    def Structure_loss(self, pred, mask):
        weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        pred = torch.sigmoid(pred)
        inter = ((pred * mask) * weit).sum(dim=(2, 3))
        union = ((pred + mask) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)
        return (wbce + wiou).mean()


if __name__ == '__main__':
    args = parser()
    trainer = Trainer(args)
    trainer.train()