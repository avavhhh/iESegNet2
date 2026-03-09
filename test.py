import torch
import torch.nn.functional as F
import numpy as np
import os
from .lib.EGANet import EGANetModel
import imageio
import torch.nn as nn
from .utils.dataloader import test_dataset
from .utils.utils import AvgMeter
from scipy.spatial.distance import directed_hausdorff
from tqdm import tqdm
import swanlab

# 1. 引入与训练同源的配置
from config import Config as cfg


def calculate_test_metrics(pred_logits, gt_mask, threshold=0.5):
    """
    计算 IoU, Pixel Accuracy (PA), Boundary F1 (BF1) 以及 Hausdorff distance (HD)
    """
    # 预测图转为概率并二值化 (与 train.py 保持完全一致，替代原版简单的 min-max 归一化)
    pred_prob = torch.sigmoid(pred_logits)
    pred_bin = (pred_prob > threshold).float()
    gt_bin = (gt_mask > 0.5).float()

    # --- 计算 IoU ---
    intersection = (pred_bin * gt_bin).sum()
    union = pred_bin.sum() + gt_bin.sum() - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)

    # --- 计算 Pixel Accuracy (PA) ---
    correct = (pred_bin == gt_bin).float()
    pa = correct.sum() / (gt_bin.shape[2] * gt_bin.shape[3])

    # --- 计算 Boundary F1 (BF1) ---
    pred_dilate = F.max_pool2d(pred_bin, kernel_size=3, stride=1, padding=1)
    pred_erode = -F.max_pool2d(-pred_bin, kernel_size=3, stride=1, padding=1)
    pred_bound = pred_dilate - pred_erode

    gt_dilate = F.max_pool2d(gt_bin, kernel_size=3, stride=1, padding=1)
    gt_erode = -F.max_pool2d(-gt_bin, kernel_size=3, stride=1, padding=1)
    gt_bound = gt_dilate - gt_erode

    bound_intersection = (pred_bound * gt_bound).sum()
    pred_bound_sum = pred_bound.sum()
    gt_bound_sum = gt_bound.sum()

    precision_b = (bound_intersection + 1e-6) / (pred_bound_sum + 1e-6)
    recall_b = (bound_intersection + 1e-6) / (gt_bound_sum + 1e-6)
    bf1 = 2 * precision_b * recall_b / (precision_b + recall_b + 1e-6)

    # --- 计算 Hausdorff Distance (HD) ---
    # 需要提取非零像素坐标，利用 scipy.spatial.distance 进行计算
    pred_bin_np = pred_bin.squeeze().cpu().numpy()
    gt_bin_np = gt_bin.squeeze().cpu().numpy()

    pred_pts = np.argwhere(pred_bin_np > 0)
    gt_pts = np.argwhere(gt_bin_np > 0)

    # 边界情况处理
    if len(pred_pts) == 0 and len(gt_pts) == 0:
        hd = 0.0
    elif len(pred_pts) == 0 or len(gt_pts) == 0:
        # 如果一方为空，惩罚距离为图像对角线长度
        hd = np.sqrt(pred_bin_np.shape[0] ** 2 + pred_bin_np.shape[1] ** 2)
    else:
        # 双向 Hausdorf 距离，取最大值
        hd1 = directed_hausdorff(pred_pts, gt_pts)[0]
        hd2 = directed_hausdorff(gt_pts, pred_pts)[0]
        hd = max(hd1, hd2)

    return iou.item(), pa.item(), bf1.item(), hd


if __name__ == '__main__':
    # ==========================
    # 1. 基础参数配置
    # ==========================
    testsize = cfg.trainsize
    pth_path = f'./checkpoints/{cfg.train_save_name}/iESegNet_latest.pth'

    # 修改点：使用相对路径，与 config.py 中的 './data/TrainDataset' 逻辑保持一致
    test_path = './data/TestDataset'
    image_root = f'{test_path}/image/'
    gt_root = f'{test_path}/mask/'

    # 结果保存路径
    save_path = f'./results/{cfg.train_save_name}/TestResult/'
    os.makedirs(save_path, exist_ok=True)

    # ==========================
    # 2. Swanlab 测试记录初始化
    # ==========================
    swanlab.init(
        project="iESegNet",
        experiment_name="DINO_Fusion_Testing",
        config={
            "testsize": testsize,
            "pth_path": pth_path,
            "image_root": image_root,
            "gt_root": gt_root
        }
    )

    print("#" * 20, "Start Testing", "#" * 20)
    print(f"[*] Loading weights from: {pth_path}")

    # 模型加载
    model = EGANetModel()
    checkpoint = torch.load(pth_path)

    # 兼容断点保存的字典格式
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.cuda()
    model.eval()

    # 初始化测试数据集加载器
    test_loader = test_dataset(image_root, gt_root, testsize)

    # 指标记录器
    iou_record = AvgMeter()
    pa_record = AvgMeter()
    bf1_record = AvgMeter()
    hd_record = AvgMeter()

    # ==========================
    # 3. 进度条与动态输出 (tqdm)
    # ==========================
    pbar = tqdm(range(test_loader.size), desc="Testing iESegNet")

    for i in pbar:
        image, gt, name = test_loader.load_data()

        # GT 预处理为 Tensor
        gt_np = np.asarray(gt, np.float32)
        gt_np /= (gt_np.max() + 1e-8)
        gt_tensor = torch.tensor(gt_np).unsqueeze(0).unsqueeze(0).cuda()

        image = image.cuda()

        with torch.no_grad():
            predicts = model(image)
            res = predicts[0]

            # 将预测结果上采样至 Ground Truth 真实尺寸用于评估计算
            res = F.upsample(res, size=gt_np.shape, mode='bilinear', align_corners=False)

            # 计算所有评估指标
            iou, pa, bf1, hd = calculate_test_metrics(res, gt_tensor)

            iou_record.update(iou, 1)
            pa_record.update(pa, 1)
            bf1_record.update(bf1, 1)
            hd_record.update(hd, 1)

            # 在进度条右侧动态显示平均 IoU 和 HD
            pbar.set_postfix({
                'IoU': f'{iou_record.show():.4f}',
                'HD': f'{hd_record.show():.2f}'
            })

            # 二值化输出保存图
            res_prob = torch.sigmoid(res).data.cpu().numpy().squeeze()
            imageio.imwrite(save_path + name, ((res_prob > 0.5) * 255).astype(np.uint8))

    # 测试结束，终端打印最终结果
    print(
        f"[*] Final Results | IoU: {iou_record.show():.4f} | PA: {pa_record.show():.4f} | BF1: {bf1_record.show():.4f} | HD: {hd_record.show():.4f}")

    # 将平均指标同步到 SwanLab 看板
    swanlab.log({
        "Test/IoU": iou_record.show(),
        "Test/Pixel_Accuracy": pa_record.show(),
        "Test/Boundary_F1": bf1_record.show(),
        "Test/Hausdorff_Distance": hd_record.show(),
    })

    print("#" * 20, "Testing Finished", "#" * 20)
