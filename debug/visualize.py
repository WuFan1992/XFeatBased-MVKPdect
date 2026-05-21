import torch
import cv2
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
        feats, kpts, reliability, vars = vudnet.net(x)
        
        heatmap = vudnet.get_kpts_heatmap(kpts)
        

    # ===== 4. 准备可视化 =====
    heatmap_vis = heatmap.squeeze().cpu().numpy()  # [H, W]
    heatmap_vis = cv2.resize(heatmap_vis, (W, H))

    rel_vis = reliability.squeeze().cpu().numpy()  # [H/8, W/8]
    rel_vis = cv2.resize(rel_vis, (W, H))

    print("heatmap range:", heatmap_vis.min(), heatmap_vis.max())
    print("reliability range:", rel_vis.min(), rel_vis.max())

    # ===== 5. 显示 =====
    plt.figure(figsize=(16, 5))

    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(img)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("VUDNet Heatmap")
    plt.imshow(heatmap_vis, cmap='hot')
    plt.colorbar()
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("VUDNet Reliability")
    plt.imshow(rel_vis, cmap='jet')
    plt.colorbar()
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    image_path = "datasets/MegaDepth_v1/0022/dense0/imgs/186069410_b743faece0_o.jpg"
    #image_path = "datasets/MegaDepth_v1/0022/dense0/imgs/217448351_09b6986ab2_o.jpg"
    run_test(image_path)