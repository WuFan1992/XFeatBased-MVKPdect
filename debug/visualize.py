import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from modules.vudnet import VUDNet


def load_image(image_path, device):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
    return img_tensor, img


def detector_map_to_heatmap(vudnet, kpts):
    """Convert detector outputs to a displayable heatmap.

    The current model may return either:
    - a single-channel detector map [B, 1, H, W]
    - coarse logits with >= 64 channels, which need VUDNet.get_kpts_heatmap
    """
    if not isinstance(kpts, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor for detector output, got {type(kpts)}")

    if kpts.dim() != 4:
        raise ValueError(f"Expected detector output with shape [B, C, H, W], got {tuple(kpts.shape)}")

    if kpts.shape[1] == 1:
        heatmap = torch.sigmoid(kpts)
    else:
        heatmap = vudnet.get_kpts_heatmap(kpts)

    return heatmap


def tensor_to_resized_map(tensor_map, out_hw, use_sigmoid=False):
    if tensor_map.dim() == 4:
        tensor_map = tensor_map[0, 0]
    elif tensor_map.dim() == 3:
        tensor_map = tensor_map[0]

    if use_sigmoid:
        tensor_map = torch.sigmoid(tensor_map)

    array = tensor_map.detach().cpu().float().numpy()
    array = cv2.resize(array, (out_hw[1], out_hw[0]))
    return array


def normalize_map(array):
    array = array.astype(np.float32)
    min_val = float(array.min())
    max_val = float(array.max())
    if max_val - min_val < 1e-8:
        return np.zeros_like(array)
    return (array - min_val) / (max_val - min_val)


def keypoints_overlay(img, keypoints, max_points=200):
    overlay = img.copy()
    if keypoints is None or len(keypoints) == 0:
        return overlay

    keypoints = np.asarray(keypoints)
    if keypoints.shape[0] > max_points:
        keypoints = keypoints[:max_points]

    for x, y in keypoints:
        cv2.circle(overlay, (int(round(x)), int(round(y))), 2, (255, 80, 80), -1)

    return overlay


def run_test(image_path, top_k=200, device="cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # ===== 1. 初始化 VUDNet =====
    vudnet = VUDNet(top_k=top_k).to(device)
    vudnet.eval()

    print(f"✅ Loaded VUDNet on device {device}")

    # ===== 2. 读取图像 =====
    img_tensor, img = load_image(image_path, device)
    H, W = img.shape[:2]

    # ===== 3. 预处理并 forward =====
    with torch.no_grad():
        x, rh, rw = vudnet.preprocess_tensor(img_tensor)
        feats, vars, match, kpts = vudnet.net(x)
        sparse = vudnet.detectAndCompute(img_tensor, top_k=top_k)[0]
        print("feats shape:", feats.shape)
        print("vars shape:", vars.shape)
        print("match shape:", match.shape)
        print("kpts shape:", kpts.shape)

        heatmap = detector_map_to_heatmap(vudnet, kpts)
        

    # ===== 4. 准备可视化 =====
    heatmap_vis = normalize_map(heatmap.squeeze().cpu().numpy())
    match_vis = normalize_map(tensor_to_resized_map(match, (H, W), use_sigmoid=True))
    var_vis = normalize_map(tensor_to_resized_map(vars, (H, W), use_sigmoid=False))
    kp_overlay = keypoints_overlay(img, sparse['keypoints'].cpu().numpy(), max_points=top_k)

    print("heatmap range:", float(heatmap_vis.min()), float(heatmap_vis.max()))
    print("match range:", float(match_vis.min()), float(match_vis.max()))
    print("variance range:", float(var_vis.min()), float(var_vis.max()))
    print("keypoints:", len(sparse['keypoints']))

    # ===== 5. 显示 =====
    fig, axes = plt.subplots(1, 5, figsize=(26, 5))

    axes[0].set_title("Original")
    axes[0].imshow(img)
    axes[0].axis('off')

    axes[1].set_title("Detector Heatmap")
    im1 = axes[1].imshow(heatmap_vis, cmap='hot')
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    axes[1].axis('off')

    axes[2].set_title("Matchability")
    im2 = axes[2].imshow(match_vis, cmap='jet')
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    axes[2].axis('off')

    axes[3].set_title("Variance")
    im3 = axes[3].imshow(var_vis, cmap='viridis')
    fig.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)
    axes[3].axis('off')

    axes[4].set_title(f"Top-{min(top_k, len(sparse['keypoints']))} Keypoints")
    axes[4].imshow(kp_overlay)
    axes[4].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    image_path = "datasets/MegaDepth_v1/0022/dense0/imgs/186069410_b743faece0_o.jpg"
    #image_path = "datasets/MegaDepth_v1/0022/dense0/imgs/217448351_09b6986ab2_o.jpg"
    run_test(image_path)