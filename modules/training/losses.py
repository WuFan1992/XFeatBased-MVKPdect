import torch
import torch.nn.functional as F

from modules.dataset.megadepth import megadepth_warper

from modules.training import utils

from third_party.alike_wrapper import extract_alike_kpts


def mv_infonce_masked(f_inv, visibility, tau=0.2):
    """
    Multi-View InfoNCE with intra-view negatives:
    - cross-view positive: corresponding points in different views
    - cross-view negative: non-corresponding points in different views  
    - intra-view negative: other points in same view (prevents feature collapse)
    
    Args:
        f_inv: [N, V, C] feature descriptors
        visibility: [N, V] visibility mask (bool)
        tau: temperature parameter
    
    Returns:
        loss: scalar loss value
    """
    N, V, C = f_inv.shape
    f = F.normalize(f_inv, dim=2)

    loss = 0.0
    count = 0

    for i in range(V):
        for j in range(i + 1, V):
            mask = visibility[:, i] & visibility[:, j]
            if mask.sum() < 10:
                continue

            fi = f[mask, i]   # [M, C]
            fj = f[mask, j]   # [M, C]

            M = fi.shape[0]

            # ===== cross-view similarity =====
            sim_cross = fi @ fj.t() / tau   # [M, M], 对角线为正样本

            # ===== intra-view similarity =====
            #sim_intra_i = fi @ fi.t() / tau
            #sim_intra_j = fj @ fj.t() / tau

            # ===== mask掉对角线（避免自己当negative）=====
            #diag_mask = torch.eye(M, device=fi.device, dtype=torch.bool)
            #sim_intra_i = sim_intra_i.masked_fill(diag_mask, float('-inf'))
            #sim_intra_j = sim_intra_j.masked_fill(diag_mask, float('-inf'))

            # ===== 拼接 logits =====
            #logits_i = torch.cat([sim_cross, sim_intra_i], dim=1)  # [M, 2M]
            #logits_j = torch.cat([sim_cross.t(), sim_intra_j], dim=1)
            
            logits_i = sim_cross
            logits_j = sim_cross.t()

            labels = torch.arange(M, device=fi.device)

            loss_i = F.cross_entropy(logits_i, labels, reduction='mean')
            loss_j = F.cross_entropy(logits_j, labels, reduction='mean')
            
            loss += loss_i + loss_j
            count += 2

    if count == 0:
        return torch.tensor(0.0, device=f.device, requires_grad=True)

    return loss / count


def dual_softmax_loss(X, Y, temp = 0.2):
    if X.size() != Y.size() or X.dim() != 2 or Y.dim() != 2:
        raise RuntimeError('Error: X and Y shapes must match and be 2D matrices')

    dist_mat = (X @ Y.t()) * temp
    conf_matrix12 = F.log_softmax(dist_mat, dim=1)
    conf_matrix21 = F.log_softmax(dist_mat.t(), dim=1)

    with torch.no_grad():
        conf12 = torch.exp( conf_matrix12 ).max(dim=-1)[0]
        conf21 = torch.exp( conf_matrix21 ).max(dim=-1)[0]
        conf = conf12 * conf21

    target = torch.arange(len(X), device = X.device)

    loss = F.nll_loss(conf_matrix12, target) + \
           F.nll_loss(conf_matrix21, target)

    return loss, conf


def alike_distill_loss(kpts, img):

    C, H, W = kpts.shape
    kpts = kpts.permute(1,2,0) 
    img = img.permute(1,2,0).expand(-1,-1,3).cpu().numpy() * 255

    with torch.no_grad():
        alike_kpts = torch.tensor( extract_alike_kpts(img), device=kpts.device )
        labels = torch.ones((H, W), dtype = torch.long, device = kpts.device) * 64 # -> Default is non-keypoint (bin 64)
        offsets = (((alike_kpts/8) - (alike_kpts/8).long())*8).long()
        offsets =  offsets[:, 0] + 8*offsets[:, 1]  # Linear IDX
        labels[(alike_kpts[:,1]/8).long(), (alike_kpts[:,0]/8).long()] = offsets

    kpts = kpts.view(-1,C)
    labels = labels.view(-1)

    mask = labels < 64
    idxs_pos = mask.nonzero().flatten()
    idxs_neg = (~mask).nonzero().flatten()
    perm = torch.randperm(idxs_neg.size(0))[:len(idxs_pos)//32]
    idxs_neg = idxs_neg[perm]
    idxs = torch.cat([idxs_pos, idxs_neg])

    kpts = kpts[idxs]
    labels = labels[idxs]

    with torch.no_grad():
        predicted = kpts.max(dim=-1)[1]
        acc =  (labels == predicted)
        acc = acc.sum() / len(acc)

    kpts = F.log_softmax(kpts, dim=1)
    
    loss = F.nll_loss(kpts, labels, reduction = 'mean')

    return loss, acc




def keypoint_loss(heatmap, target):
    # Compute L1 loss
    target = target.unsqueeze(1)
    L1_loss = F.l1_loss(heatmap, target)
    return L1_loss * 3.0

def compute_p_correct(f_inv, visibility, tau=0.1, eps=1e-6):
    """
    f_inv: [N, V, C]
    visibility: [N, V]
    returns: [N, V] p_correct
    """
    N, V, C = f_inv.shape
    f = F.normalize(f_inv, dim=2)  # [N,V,C]

    p_all = torch.zeros(N, V, device=f.device)
    count = torch.zeros(N, V, device=f.device)

    for i in range(V):
        for j in range(i + 1, V):
            mask = visibility[:, i] & visibility[:, j]  # [N]
            idx = mask.nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() < 10:
                continue

            fi = f[idx, i]  # [M, C]
            fj = f[idx, j]  # [M, C]

            # [M, M] 相似度矩阵
            sim = fi @ fj.t() / tau
            prob = F.softmax(sim, dim=1)
            p = torch.diagonal(prob)  # [M], 正确匹配概率

            # 累加
            p_all[idx, i] += p
            p_all[idx, j] += p
            count[idx, i] += 1
            count[idx, j] += 1

    p_all = p_all / (count + eps)
    return p_all.clamp(eps, 1.0)

def sigma_loss_from_pcorrect(f_inv, sigma_pred, visibility):

    p_correct = compute_p_correct(f_inv, visibility)  # [N,V]

    var_target = (1.0 - p_correct).detach()

    sigma_pred = sigma_pred.squeeze(-1)

    loss = F.mse_loss(sigma_pred, var_target)

    return loss, var_target



def multi_view_xfeat_heatmap_loss(
    hmaps,
    feats,
    visibility,
    tau=0.2,
    eps=1e-6
    ):
    """
    Multi-view XFeat-style heatmap supervision.

    ```
    Args:
        hmaps: [N, V, 1]
        feats: [N, V, C]
        visibility: [N, V]

    Returns:
        loss
    """

    N, V, C = feats.shape

    feats = F.normalize(feats, dim=-1)

    total_loss = 0.0
    total_pairs = 0

    for i in range(V):

        for j in range(i + 1, V):

            mask = visibility[:, i] & visibility[:, j]

            idx = mask.nonzero(as_tuple=False).squeeze(-1)

            if len(idx) < 10:
                continue

            fi = feats[idx, i]
            fj = feats[idx, j]

            # similarity matrix
            sim = (fi @ fj.t()) / tau

            # dual softmax
            prob12 = F.softmax(sim, dim=1)
            prob21 = F.softmax(sim.t(), dim=1)

            # XFeat confidence
            conf12 = prob12.max(dim=1)[0]
            conf21 = prob21.max(dim=1)[0]

            conf = (conf12 * conf21).detach()

            hi = torch.sigmoid(hmaps[idx, i, 0])
            hj = torch.sigmoid(hmaps[idx, j, 0])

            # supervise BOTH views
            loss_i = F.l1_loss(hi, conf)
            loss_j = F.l1_loss(hj, conf)

            total_loss += loss_i + loss_j
            total_pairs += 2

    if total_pairs == 0:
        return torch.tensor(
            0.0,
            device=feats.device,
            requires_grad=True
        )

    return total_loss / total_pairs




