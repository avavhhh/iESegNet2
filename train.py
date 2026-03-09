import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import os
from .lib.EGANet import EGANetModel
from .utils.dataloader import PolypDataset, get_loader, test_dataset
from .utils.loss import DeepSupervisionLoss
from .utils.utils import AvgMeter, clip_gradient
from datetime import datetime
from torch.optim.lr_scheduler import MultiStepLR, LambdaLR
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
import time
import swanlab

# 引入解耦后的配置文件
from config import Config as cfg


# 指标计算
def calculate_metrics(pred_logits, gt_mask, threshold=0.5):
    """
    计算 IoU, Pixel Accuracy (PA), Boundary F1 (BF1)
    利用 GPU 张量计算以保证训练速度
    """
    # 1. 预测图转为概率 (Sigmoid) 并进行二值化
    pred_prob = torch.sigmoid(pred_logits)
    pred_bin = (pred_prob > threshold).float()
    gt_bin = (gt_mask > 0.5).float()

    # --- 计算 IoU ---
    intersection = (pred_bin * gt_bin).sum(dim=(2, 3))
    union = pred_bin.sum(dim=(2, 3)) + gt_bin.sum(dim=(2, 3)) - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)

    # --- 计算 Pixel Accuracy (PA) ---
    correct = (pred_bin == gt_bin).float()
    pa = correct.sum(dim=(2, 3)) / (gt_bin.shape[2] * gt_bin.shape[3])

    # --- 计算 Boundary F1 (BF1) ---
    # 利用 MaxPool 实现形态学边界提取: 边界 = 膨胀图 - 腐蚀图
    pred_dilate = F.max_pool2d(pred_bin, kernel_size=3, stride=1, padding=1)
    pred_erode = -F.max_pool2d(-pred_bin, kernel_size=3, stride=1, padding=1)
    pred_bound = pred_dilate - pred_erode

    gt_dilate = F.max_pool2d(gt_bin, kernel_size=3, stride=1, padding=1)
    gt_erode = -F.max_pool2d(-gt_bin, kernel_size=3, stride=1, padding=1)
    gt_bound = gt_dilate - gt_erode

    # 计算边界的 Precision 和 Recall
    bound_intersection = (pred_bound * gt_bound).sum(dim=(2, 3))
    pred_bound_sum = pred_bound.sum(dim=(2, 3))
    gt_bound_sum = gt_bound.sum(dim=(2, 3))

    precision_b = (bound_intersection + 1e-6) / (pred_bound_sum + 1e-6)
    recall_b = (bound_intersection + 1e-6) / (gt_bound_sum + 1e-6)

    bf1 = 2 * precision_b * recall_b / (precision_b + recall_b + 1e-6)

    return iou.mean(), pa.mean(), bf1.mean()


# 功能2修改：在此处增加 scheduler 参数，为了能在保存断点时保存学习率状态
def train(train_loader, model, optimizer, scheduler, epoch, criteria_loss, writer, device, total_step):
    model.train()  # DINO在EGANet中重写了train，会自动维持在eval()

    # ---- multi-scale training ----
    size_rates = cfg.size_rates
    # 实例化指标记录器
    loss_record = AvgMeter()
    iou_record = AvgMeter()
    pa_record = AvgMeter()
    bf1_record = AvgMeter()

    # 功能1：加入进度条 (tqdm)，显示当前进度和预估时间
    pbar = tqdm(train_loader, total=total_step, desc=f"Epoch [{epoch:03d}/{cfg.epoch:03d}]")
    start_time = time.time()  # 记录 Epoch 开始时间

    for i, pack in enumerate(pbar, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            # ---- data prepare ----
            images, gts = pack
            images = Variable(images).to(device)
            gts = Variable(gts).to(device)

            # 使用配置文件的 trainsize
            trainsize = int(round(cfg.trainsize * rate / 32) * 32)
            if rate != 1:
                images = F.interpolate(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.interpolate(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)

            predicts = model(images)
            loss = criteria_loss(predicts, gts)

            writer.add_scalar("Loss/train", loss.item(), epoch * total_step + i)

            loss.backward()
            clip_gradient(optimizer, cfg.grad_norm)
            optimizer.step()

            if rate == 1:
                loss_record.update(loss.item(), cfg.batchsize)

                # ==== 提取主输出特征图计算各项评估指标 ====
                main_pred = predicts[0].detach()  # detach() 取消梯度追踪，避免内存泄漏
                iou, pa, bf1 = calculate_metrics(main_pred, gts)

                iou_record.update(iou.item(), cfg.batchsize)
                pa_record.update(pa.item(), cfg.batchsize)
                bf1_record.update(bf1.item(), cfg.batchsize)

        # 获取当前的学习率
        current_lr = optimizer.param_groups[0]['lr']

        # 功能1：在进度条右侧动态显示当前损失、学习率等详细指标参数
        pbar.set_postfix({
            'Loss': f'{loss_record.show():.4f}',
            'IoU': f'{iou_record.show():.4f}',
            'PA': f'{pa_record.show():.4f}',
            'LR': f'{current_lr:.6f}'
        })

        # --- SwanLab 记录：Step 级别信息 ---
        if i % cfg.print_freq == 0 or i == total_step:
            swanlab.log({
                "Train/Loss": loss_record.show(),
                "Train_Step/IoU": iou_record.show(),
                "Train_Step/Pixel_Accuracy": pa_record.show(),
                "Train_Step/Boundary_F1": bf1_record.show(),
                "Train/Learning_Rate": current_lr,
                "Train_Step/Epoch": epoch,
            }, step=epoch * total_step + i)

    # 功能1：计算当前 Epoch 所花费的时间，并用 tqdm.write 替代 print 避免破坏进度条排版
    epoch_time = time.time() - start_time
    # --- SwanLab 记录：Epoch 级别汇总信息 ---
    swanlab.log({
        "Train_Epoch/Loss": loss_record.show(),
        "Train_Epoch/IoU": iou_record.show(),
        "Train_Epoch/Pixel_Accuracy": pa_record.show(),
        "Train_Epoch/Boundary_F1": bf1_record.show(),
        "Train_Epoch/Time_s": epoch_time,
    }, step=epoch)
    tqdm.write(f"[*] {datetime.now()} Epoch [{epoch:03d}/{cfg.epoch:03d}] Finished. Time: {epoch_time:.2f}s "
               f"| Loss: {loss_record.show():.4f} | IoU: {iou_record.show():.4f} "
               f"| PA: {pa_record.show():.4f} | BF1: {bf1_record.show():.4f}")

    # 获取保存路径
    save_path = cfg.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 功能2：断点保存功能。每个 Epoch 结束后更新 latest_checkpoint
    # 保存所有的网络状态，确保无论因为什么原因暂停，都可以无缝衔接
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss_record.show()
    }
    latest_save_file = os.path.join(save_path, 'iESegNet2_latest.pth')
    torch.save(checkpoint, latest_save_file)

    # 正常按频率保存纯模型权重（用于后续测试和推理）
    if (epoch + 1) % cfg.save_freq == 0:
        save_file = os.path.join(save_path, 'iESegNet2-%d.pth' % epoch)
        torch.save(model.state_dict(), save_file)
        tqdm.write(f'[Saving Snapshot:] {save_file}')


if __name__ == '__main__':
    # 功能3：SwanLab 初始化，将 config 中的重要超参数同步到后台
    swanlab.init(
        project="iESegNet2",
        experiment_name="iESegNet2_Training",
        config={
            "epoch": cfg.epoch,
            "lr": cfg.lr,
            "batchsize": cfg.batchsize,
            "trainsize": cfg.trainsize,
            "size_rates": cfg.size_rates,
            "grad_norm": cfg.grad_norm,
            "weight_decay": cfg.weight_decay,
            "momentum": cfg.momentum
        }
    )

    writer = SummaryWriter(log_dir=cfg.log_dir)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = EGANetModel()

    if cfg.mgpu and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model.to(device)

    # 优化器只传入 requires_grad = True 的参数，完美排除DINO被优化
    params = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = torch.optim.SGD(params, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)

    lr_lambda = lambda epoch: 1.0 - pow((epoch / cfg.epoch), cfg.power)
    scheduler = LambdaLR(optimizer, lr_lambda)

    criteria_loss = DeepSupervisionLoss(typeloss="StructureLoss")

    image_root = '{}/image/'.format(cfg.train_path)
    gt_root = '{}/mask/'.format(cfg.train_path)

    train_loader = get_loader(image_root, gt_root, batchsize=cfg.batchsize, trainsize=cfg.trainsize)
    total_step = len(train_loader)

    if torch.cuda.is_available():
        print("Device:", torch.cuda.get_device_name(0))
    print("#" * 20, "Start Training", "#" * 20)

    pytorch_total_params = sum(p.numel() for p in filter(lambda p: p.requires_grad, model.parameters()))
    print(f"Trainable Parameters: {pytorch_total_params}")

    # ================= 功能2：断点续接功能 =================
    start_epoch = 0
    latest_save_file = os.path.join(cfg.save_path, 'iESegNet2_latest.pth')

    if os.path.exists(latest_save_file):
        print(f"[*] Found checkpoint at {latest_save_file}, loading to resume training...")
        checkpoint = torch.load(latest_save_file, map_location=device)
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"[*] Successfully resumed from epoch {checkpoint['epoch']}. Next up: epoch {start_epoch}")
    # =======================================================

    for epoch in range(start_epoch, cfg.epoch):
        # 增加传入 scheduler 以供在内部保存调度器状态
        train(train_loader, model, optimizer, scheduler, epoch, criteria_loss, writer, device, total_step)
        scheduler.step()

    writer.close()
