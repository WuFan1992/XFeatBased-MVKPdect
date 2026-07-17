from modules.vudnet import *
import cv2
import matplotlib.pyplot as plt

from validation.utils import *

def visualize_vudnet_matches(img1, img2, mkpts1, mkpts2):
    """
    img1, img2: numpy (H,W,3)
    mkpts1, mkpts2: (N,2)
    """

    # 👉 padding（和你之前一致）
    def pad_to_same_height(img1, img2):
        h1, w1, _ = img1.shape
        h2, w2, _ = img2.shape
        max_h = max(h1, h2)

        def pad(img, target_h):
            h, w, c = img.shape
            pad_h = target_h - h
            return np.pad(img, ((0,pad_h),(0,0),(0,0)), mode='constant')

        return pad(img1, max_h), pad(img2, max_h)

    img1, img2 = pad_to_same_height(img1, img2)

    concat_img = np.concatenate([img1, img2], axis=1)

    plt.figure(figsize=(15,7))
    plt.imshow(concat_img)

    for i in range(len(mkpts1)):
        pt1 = mkpts1[i]
        pt2 = mkpts2[i] + np.array([img1.shape[1], 0])

        plt.plot([pt1[0], pt2[0]],
                 [pt1[1], pt2[1]],
                 'r', linewidth=1)

    plt.title(f"VUDNet Matches: {len(mkpts1)}")
    plt.axis('off')
    plt.show()


class VUDNet_helper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.eval().cuda()

    def load_image(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def match(self, img1_path, img2_path, top_k=4096):
        
        img1 = self.load_image(img1_path)
        img2 = self.load_image(img2_path)

        # 👉 VUDNet 通常自带 detect + describe + match
        kpts0, kpts1, sigma0, sigma1 = self.model.match_vudnet(img1, img2, top_k=top_k)
       
        # ✅ 和你原接口对齐
        if len(kpts0) == 0:
            return np.zeros((0,2)), np.zeros((0,2)), np.array([]), np.array([])
        

        # Debug
        #visualize_vudnet_matches(img1, img2, kpts0, kpts1)

        return kpts0, kpts1, sigma0, sigma1