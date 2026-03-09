import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
import argparse
from PIL import Image
import torchvision.transforms as transforms
from .lib.EGANet import EGANetModel


def process_image(img_path, img_size=352):
    """
    读取并预处理图像
    """
    image = Image.open(img_path).convert('RGB')
    original_size = image.size  # (Width, Height)

    # MEGANet 通常使用 ImageNet 的标准归一化参数
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    input_tensor = transform(image).unsqueeze(0)  # 增加 Batch 维度: [1, 3, H, W]
    return input_tensor, original_size, np.array(image)


def apply_mask_overlay(image_np, mask_np, alpha=0.5, color=(0, 255, 0)):
    """
    将 mask 叠加在原图上 (默认使用绿色半透明叠加)
    image_np: RGB 原图 (H, W, 3)
    mask_np: 二值化 Mask (H, W) 取值为 0 或 255
    """
    # 将 mask 转换为彩色图 (H, W, 3)
    color_mask = np.zeros_like(image_np)
    # 将 mask 的 255 区域赋予指定的颜色 (这里设为绿色)
    color_mask[mask_np == 255] = color

    # 生成混合图像
    overlay_img = cv2.addWeighted(image_np, 1.0, color_mask, alpha, 0)
    return overlay_img


def main():
    parser = argparse.ArgumentParser(description="MEGANet-DINO Demo")
    parser.add_argument('--image_path', type=str, required=True, help='输入图像的路径')
    parser.add_argument('--weight_path', type=str, default='./checkpoints/MEGANet-Res2Net/MEGANet_latest.pth',
                        help='模型权重路径')
    parser.add_argument('--save_path', type=str, default='./demo_result.jpg', help='结果保存路径')
    parser.add_argument('--img_size', type=int, default=352, help='模型输入图像尺寸')
    opt = parser.parse_args()

    # 1. 检查设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] 使用设备: {device}")

    # 2. 实例化模型并加载权重
    print("[*] 正在加载模型和权重...")
    model = EGANetModel(n_channels=3, n_classes=1)

    if not os.path.exists(opt.weight_path):
        raise FileNotFoundError(f"找不到权重文件: {opt.weight_path}")

    # 因为保存断点时可能保存的是字典，这里做个兼容处理
    checkpoint = torch.load(opt.weight_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # strict=False 防止冻结的 DINO 权重导致的 key 匹配问题
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    # 3. 数据预处理
    print(f"[*] 正在处理图像: {opt.image_path}")
    input_tensor, (orig_w, orig_h), orig_img_np = process_image(opt.image_path, opt.img_size)
    input_tensor = input_tensor.to(device)

    # 4. 模型推理
    print("[*] 正在进行前向推理...")
    with torch.no_grad():
        # EGANet 会输出 5 个尺度的特征，out1 是最终输出
        predicts = model(input_tensor)
        res = predicts[0]

    # 5. 后处理与上采样回原图尺寸
    res = F.interpolate(res, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
    res = res.data.cpu().numpy().squeeze()  # 取出特征图数据并去除多余维度: (H, W)

    # 归一化到 0~1
    res = (res - res.min()) / (res.max() - res.min() + 1e-8)

    # 二值化处理 (阈值通常设为 0.5)，并转换为 0 或 255 的图像
    mask = ((res > 0.5) * 255).astype(np.uint8)

    # 6. 生成可视化图像
    print("[*] 正在生成叠加结果并保存...")
    # BGR 转换处理，因为 OpenCV 保存图片默认通道是 BGR
    orig_img_bgr = cv2.cvtColor(orig_img_np, cv2.COLOR_RGB2BGR)

    # 获得半透明叠加结果
    overlay_img_bgr = apply_mask_overlay(orig_img_bgr, mask, alpha=0.6, color=(0, 255, 0))  # 绿色叠加

    # 将单通道的 Mask 转换为三通道，以便后续拼接
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # 将 [原图, 纯Mask图, 叠加结果图] 横向拼接
    concat_result = np.hstack((orig_img_bgr, mask_bgr, overlay_img_bgr))

    # 保存结果
    cv2.imwrite(opt.save_path, concat_result)
    print(f"[*] 推理完成！结果已保存至: {opt.save_path}")


if __name__ == "__main__":
    main()
