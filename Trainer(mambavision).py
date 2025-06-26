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
from Network.Network_MambaVision import FMFS  # MambaVision



def parser():
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument('--batchsize', default=10, type=int, help="Batch size for training")
    parser.add_argument('--savepath', default="../FMFS-Net/checkpoints", type=str, help="Path to save checkpoints")
    parser.add_argument('--datapath', default="../FMFS-Net/data", type=str, help="Path to dataset")
    parser.add_argument('--logpath', default="../FMFS-Net/logs", type=str, help="Path to logs")
    parser.add_argument('--checkpoint', default='../FMFS-Net/checkpoints/Net_epoch_best.pth', type=str, help="Path to pre-trained checkpoint")  # mambavision
    parser.add_argument('--val_pred', default='../FMFS-Net/val_pred', type=str, help="Path to save validation predictions")
    parser.add_argument('--lr', default=1e-5, type=float, help="Learning rate")
    parser.add_argument('--epoch', default=40, type=int, help="Number of epochs")
    parser.add_argument('--device', default="cuda:0" ,type=str, help="Device (cuda or cpu)")
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
        self.warmup_scheduler = LinearLR(self.optimizer, start_factor=0.1, end_factor=1.0, total_iters=5)
        self.main_scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=3)
        self.use_warmup = True  # 控制是否处于 warmup 阶段，断点启动时关掉

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

        # 初始化用于存储每个epoch的损失值
        self.epoch_losses = {'total_loss': [], 'loss_1': [], 'loss_2': [], 'loss_3': [], 'loss_4': [], 'loss_5': []}

        # 检查点恢复
        if args.checkpoint:
            self.use_warmup = False
            self.model.load_state_dict(torch.load(args.checkpoint))
            print(f"Loaded checkpoint from {args.checkpoint}")

        # 验证集预测图保存路径
        self.val_pred = args.val_pred
        os.makedirs(self.val_pred, exist_ok=True)


    def train(self):
        for epoch in range(1, self.args.epoch + 1):
            start_time = datetime.now()

            if self.use_warmup and epoch <= 5:
                print(f"Warmup phase: Epoch {epoch}")
                avg_loss, avg_parts = self.train_epoch(epoch)
                self.warmup_scheduler.step()
            else:
                self.use_warmup = False
                avg_loss, avg_parts = self.train_epoch(epoch)
                val_loss, eval_mae, eval_Fm, eval_Sm, eval_Em, eval_IoU = self.validate(current_epoch=epoch)
                self.main_scheduler.step(val_loss)


                cur_score = eval_mae - eval_Fm - eval_Sm - eval_Em

                log_message = (
                    f"Epoch {epoch}/{self.args.epoch} | "
                    f"Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f} | "
                    # f"MAE: {eval_mae:.4f} | Fm: {eval_Fm:.4f} | Sm: {eval_Sm:.4f} | Em: {eval_Em:.4f} | IoU: {eval_IoU:.4f} | "
                    f"Eval: {cur_score:.4f}"
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
                for i, loss_val in enumerate(avg_parts, 1):
                    self.writer.add_scalar(f"Loss/train_part_{i}", loss_val, epoch)

                # 记录损失
                self.epoch_losses['total_loss'].append(avg_loss)
                for idx, loss_val in enumerate(avg_parts):
                    self.epoch_losses[f'loss_{idx + 1}'].append(loss_val)

                if epoch >= 1 and cur_score < self.best_score:      # 断点时修改
                    self.best_score = cur_score
                    self.best_epoch = epoch
                    torch.save(self.model.state_dict(), os.path.join(self.args.savepath, 'Net_epoch_best.pth'))  # mambavision
                    print(f">>> Save state_dict successfully! Best epoch is {epoch}.")

            if epoch >= 1:  # 断点时修改
                # 每10个epoch保存一次权重
                if epoch % 10 == 0:
                    save_name = f'Net_epoch_{epoch}.pth'  # mambavision
                    torch.save(self.model.state_dict(), os.path.join(self.args.savepath, save_name))
                    print(f">>> Save state_dict successfully! Epoch is {epoch}.")

                # 验证并保存预测结果
                self.validate(current_epoch=epoch)

            end_time = datetime.now()
            duration = end_time - start_time
            logging.info(f"Epoch {epoch} finished. Duration: {duration}, Total Loss: {avg_loss:.4f}")


    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        total_parts = [0.0] * 5  # loss_1 ~ loss_5

        loop = tqdm(self.train_loader, desc=f"Epoch {epoch}", leave=True)

        for images, masks in loop:
            images = images.to(self.device, dtype=torch.float32)
            masks = masks.to(self.device, dtype=torch.float32)

            outputs = self.model(images)
            p1, p2, p3, p4, p5 = outputs


            target_size = masks.shape[-2:]
            p1 = F.interpolate(p1, size=target_size, mode='bilinear', align_corners=False)
            p2 = F.interpolate(p2, size=target_size, mode='bilinear', align_corners=False)
            p3 = F.interpolate(p3, size=target_size, mode='bilinear', align_corners=False)
            p4 = F.interpolate(p4, size=target_size, mode='bilinear', align_corners=False)
            p5 = F.interpolate(p5, size=target_size, mode='bilinear', align_corners=False)

            loss, l1, l2, l3, l4, l5  = self.Loss_Function((p1, p2, p3, p4, p5), masks)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_parts[0] += l1
            total_parts[1] += l2
            total_parts[2] += l3
            total_parts[3] += l4
            total_parts[4] += l5
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(self.train_loader)
        avg_parts = [p / len(self.train_loader) for p in total_parts]

        return avg_loss, avg_parts

    def validate(self, current_epoch):
        metric = Metric()
        total_loss = 0.0
        self.model.eval()

        with torch.no_grad():
            for idx, (images, masks) in enumerate(self.val_loader):
                images = images.to(self.device, dtype=torch.float32)
                masks = masks.to(self.device, dtype=torch.float32)

                outputs = self.model(images)
                p1 = outputs if not self.model.training else outputs[0]

                # 输出验证集前5张图像预测图
                target_size = masks.shape[-2:]
                res = F.interpolate(p1, size=target_size, mode='bilinear', align_corners=False)
                res = res.sigmoid().cpu().numpy()  # [B, H, W], range [0,1]
                masks_np = masks.cpu().numpy()  # [B, H, W], range [0,1]

                if idx < 5:  # 只保存前五张验证图像的预测结果
                    for pred_i, gt_i in zip(res, masks_np):
                        pred_uint8 = (pred_i.squeeze() * 255).astype(np.uint8)
                        gt_uint8 = (gt_i.squeeze() > 0.5).astype(np.uint8) * 255

                        # 保存预测与GT拼接图
                        save_dir = os.path.join(self.val_pred, f"{current_epoch}_epoch")
                        os.makedirs(save_dir, exist_ok=True)
                        save_img_path = os.path.join(save_dir, f"pred_gt_pair_{idx}.png")
                        cv2.imwrite(save_img_path, np.hstack([pred_uint8, gt_uint8]))

                        metric.step(pred=pred_uint8, gt=gt_uint8)

                loss = self.Structure_loss(p1, masks).item()
                total_loss += loss

            results = metric.get_results(bit_width=4)
            avg_loss = total_loss / len(self.val_loader)

            log_str = f"Epoch {current_epoch}/100 | "
            log_str += f"Val Loss: {avg_loss:.4f} | "
            log_str += " | ".join([f"{k}: {v}" for k, v in results.items()])
            print(log_str)

            return avg_loss, float(results["MAE"]), float(results["meanFm"]), float(results["Smeasure"]), float(
                results["meanEm"]), float(results["Mean_IoU"])


    def Loss_Function(self, preds, mask):

        p1, p2, p3, p4, p5 = preds

        loss_1 = self.Structure_loss(p1, mask)
        loss_2 = self.Structure_loss(p2, mask)
        loss_3 = self.Structure_loss(p3, mask)
        loss_4 = self.Structure_loss(p4, mask)
        loss_5 = self.Structure_loss(p5, mask)

        total_loss = (loss_1 + 0.6 * loss_2 + 0.4 * loss_3 + 0.3 * loss_4 + 0.5 * loss_5)

        return total_loss, loss_1, loss_2, loss_3, loss_4, loss_5

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