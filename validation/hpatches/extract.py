import os
import sys
import cv2
from pathlib import Path
import numpy as np
import torch
import torch.utils.data as data
from tqdm import tqdm
from modules.vudnet import *

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


dataset_root = 'datasets/hseq/hpatches-sequences-release'
use_cuda = torch.cuda.is_available()
device = 'cuda' if use_cuda else 'cpu'
methods = ['alike-n', 'alike-l', 'alike-n-ms', 'alike-l-ms']


class HPatchesDataset(data.Dataset):
    def __init__(self, root: str = dataset_root, alteration: str = 'all'):  # alteration 控制读取哪些序列，是全部，还是只有光照变化，又或是只有view 变化
        """
        Args:
            root: dataset root path
            alteration: # 'all', 'i' for illumination or 'v' for viewpoint
        """
        assert (Path(root).exists()), f"Dataset root path {root} dose not exist!"
        self.root = root

        # get all image file name
        self.image0_list = []
        self.image1_list = []
        self.homographies = []
        folders = [x for x in Path(self.root).iterdir() if x.is_dir()] # [v_indiana, i_bricks ....]
        self.seqs = []
        for folder in folders:
            if alteration == 'i' and folder.stem[0] != 'i':
                continue
            if alteration == 'v' and folder.stem[0] != 'v':
                continue

            self.seqs.append(folder)  

        self.len = len(self.seqs)
        assert (self.len > 0), f'Can not find PatchDataset in path {self.root}'

    def __getitem__(self, item):
        folder = self.seqs[item]

        imgs = []
        homos = []
        for i in range(1, 7):
            img = cv2.imread(str(folder / f'{i}.ppm'), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # HxWxC
            imgs.append(img)

            if i != 1:
                homo = np.loadtxt(str(folder / f'H_1_{i}')).astype('float32')
                homos.append(homo)

        return imgs, homos, folder.stem

    def __len__(self):
        return self.len

    def name(self):
        return self.__class__


def extract_vudnet(model, img, top_k=4096):

    img = torch.tensor(
        img.astype(np.float32) / 255.
    ).permute(2,0,1).unsqueeze(0)

    pred = model.detectAndCompute(
        img,
        top_k=top_k
    )[0]

    return {
        'keypoints': pred['keypoints'],
        'descriptors': pred['descriptors'],
        'scores': pred['scores']
    }

def extract_method(method):

    hpatches = HPatchesDataset(
        root=dataset_root,
        alteration='all'
    )

    model = VUDNet(
    ).eval()

    progbar = tqdm(
        hpatches,
        desc=f'Extracting {method}'
    )

    for imgs, homos, seq_name in progbar:

        for i in range(1,7):

            img = imgs[i-1]

            pred = extract_vudnet(
                model,
                img,
                top_k=5000
            )

            kpts = pred['keypoints']
            descs = pred['descriptors']
            scores = pred['scores']

            save_path = os.path.join(
                dataset_root,
                seq_name,
                f'{i}.ppm.{method}'
            )

            with open(save_path, 'wb') as f:

                np.savez(
                    f,
                    keypoints=kpts.cpu().numpy(),
                    descriptors=descs.cpu().numpy(),
                    scores=scores.cpu().numpy()
                )


if __name__ == '__main__':
    for method in methods:
        extract_method(method)