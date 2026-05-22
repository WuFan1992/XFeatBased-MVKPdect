import cv2
import numpy as np
import h5py
import torch
import torch.nn.functional as F



def imread_gray(path, augment_fn=None):
    
    image = cv2.imread(str(path), 1)

    if augment_fn is not None:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = augment_fn(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image  # (h, w)


def get_resized_wh(w, h, resize=None):
    if resize is not None:  # resize the longer edge
        scale = resize / max(h, w)
        w_new, h_new = int(round(w*scale)), int(round(h*scale))
    else:
        w_new, h_new = w, h
    return w_new, h_new


def get_divisible_wh(w, h, df=None):
    if df is not None:
        w_new, h_new = map(lambda x: int(x // df * df), [w, h])
    else:
        w_new, h_new = w, h
    return w_new, h_new


def pad_bottom_right(inp, pad_size, ret_mask=False):
    assert isinstance(pad_size, int) and pad_size >= max(inp.shape[-2:]), f"{pad_size} < {max(inp.shape[-2:])}"
    mask = None
    if inp.ndim == 2:
        padded = np.zeros((pad_size, pad_size), dtype=inp.dtype)
        padded[:inp.shape[0], :inp.shape[1]] = inp
        if ret_mask:
            mask = np.zeros((pad_size, pad_size), dtype=bool)
            mask[:inp.shape[0], :inp.shape[1]] = True
    elif inp.ndim == 3:
        padded = np.zeros((inp.shape[0], pad_size, pad_size), dtype=inp.dtype)
        padded[:, :inp.shape[1], :inp.shape[2]] = inp
        if ret_mask:
            mask = np.zeros((inp.shape[0], pad_size, pad_size), dtype=bool)
            mask[:, :inp.shape[1], :inp.shape[2]] = True
    else:
        raise NotImplementedError()
    return padded, mask


# --- MEGADEPTH ---

def fix_path_from_d2net(path):
    if not path:
        return None

    path = path.replace('Undistorted_SfM/', '')
    path = path.replace('images', 'dense0/imgs')
    path = path.replace('phoenix/S6/zl548/MegaDepth_v1/', '')

    return path

def read_megadepth_gray(path, resize=None, df=None, padding=False, augment_fn=None):
    """
    Args:
        resize (int, optional): the longer edge of resized images. None for no resize.
        padding (bool): If set to 'True', zero-pad resized images to squared size.
        augment_fn (callable, optional): augments images with pre-defined visual effects
    Returns:
        image (torch.tensor): (1, h, w)
        mask (torch.tensor): (h, w)
        scale (torch.tensor): [w/w_new, h/h_new]        
    """
    # read image
    image = imread_gray(path, augment_fn)

    # resize image
    w, h = image.shape[1], image.shape[0]

    if len(resize) == 2:
        w_new, h_new = resize
    else:
        resize = resize[0]
        w_new, h_new = get_resized_wh(w, h, resize)
        w_new, h_new = get_divisible_wh(w_new, h_new, df)


    image = cv2.resize(image, (w_new, h_new))
    scale = torch.tensor([w/w_new, h/h_new], dtype=torch.float)

    if padding:  # padding
        pad_to = max(h_new, w_new)
        image, mask = pad_bottom_right(image, pad_to, ret_mask=True)  # padding 的区域 mask 是 0
    else:
        mask = None

    #image = torch.from_numpy(image).float()[None] / 255  # (h, w) -> (1, h, w) and normalized
    image = torch.from_numpy(image).float().permute(2,0,1) / 255  # (h, w) -> (1, h, w) and normalized
    mask = torch.from_numpy(mask) if mask is not None else None

    return image, mask, scale


def read_megadepth_depth(path, pad_to=None):
   
    depth = np.array(h5py.File(path, 'r')['depth'])
    if pad_to is not None:
        depth, _ = pad_bottom_right(depth, pad_to, ret_mask=False)
    depth = torch.from_numpy(depth).float()  # (h, w)
    return depth



def sample_map_at_coords(fmap, coords, H, W, mode='bilinear'):
    """
    fmap:   [1, C, Hc, Wc]
    coords: [N, 2]  (原图坐标)
    H, W:   原图尺寸
    mode:   'bilinear' or 'nearest'
    """

    device = fmap.device
    coords = coords.to(device).float()

    # ===== ⭐ 1. 先缩放到 feature map 尺度 =====
    _, _, Hc, Wc = fmap.shape
    coords = coords.clone()
    coords[..., 0] = coords[..., 0] * (Wc / W)
    coords[..., 1] = coords[..., 1] * (Hc / H)

    if mode == 'nearest':
        coords_nn = coords.round().long()
        coords_nn[..., 0] = coords_nn[..., 0].clamp(0, Wc - 1)
        coords_nn[..., 1] = coords_nn[..., 1].clamp(0, Hc - 1)
        sampled = fmap[0, :, coords_nn[:, 1], coords_nn[:, 0]]  # [C, N]
        return sampled.transpose(0, 1)

    # ===== 2. 再 normalize 到 [-1,1] =====
    coords_norm = coords.clone()
    coords_norm[..., 0] = coords_norm[..., 0] / (Wc - 1) * 2 - 1
    coords_norm[..., 1] = coords_norm[..., 1] / (Hc - 1) * 2 - 1

    # ===== 3. reshape =====
    coords_norm = coords_norm.unsqueeze(0).unsqueeze(2)  # [1, N, 1, 2]

    # ===== 4. grid_sample =====
    sampled = F.grid_sample(
        fmap, coords_norm,
        mode=mode,
        align_corners=True
    )  # [1, C, N, 1]

    # ===== 5. reshape → [N,C] =====
    sampled = sampled.squeeze(0).squeeze(-1).transpose(0, 1)

    return sampled