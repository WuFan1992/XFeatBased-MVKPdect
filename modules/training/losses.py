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


def dual_softmax_loss(X, Y, temp = 0.2, hard_neg_X=None, hard_neg_Y=None, hard_neg_weight=0.3, margin=0.1):
    """
    Dual softmax loss with optional hard negative mining from multi-view data.
    
    Args:
        X, Y: [M, C] matched feature descriptors from two views
        temp: temperature for softmax
        hard_neg_X: [M, K, C] hard negative features for view X (from other views of same 3D point)
        hard_neg_Y: [M, K, C] hard negative features for view Y
        hard_neg_weight: weight for hard negative loss component
        margin: margin for triplet loss
    
    Returns:
        loss: scalar loss value
        conf: [M] confidence scores
    """
    if X.size() != Y.size() or X.dim() != 2 or Y.dim() != 2:
        raise RuntimeError('Error: X and Y shapes must match and be 2D matrices')

    # ===== Standard pair-wise loss =====
    dist_mat = (X @ Y.t()) * temp
    conf_matrix12 = F.log_softmax(dist_mat, dim=1)
    conf_matrix21 = F.log_softmax(dist_mat.t(), dim=1)

    target = torch.arange(len(X), device = X.device)

    pair_loss = F.nll_loss(conf_matrix12, target) + \
                F.nll_loss(conf_matrix21, target)

    # ===== Hard negative loss (multi-view) =====
    hard_loss = torch.tensor(0.0, device=X.device, requires_grad=True)
    
    if hard_neg_X is not None and hard_neg_Y is not None:
        # hard_neg_X: [M, K, C] - K hard negatives for each matched point
        M, K, C = hard_neg_X.shape
        
        # Positive similarity (diagonal of dist_mat)
        pos_sim = (X * Y).sum(dim=1)  # [M]
        
        # Hard negative similarity
        # Compute X @ hard_neg_X^T using einsum: [M, K]
        hard_sim_X = torch.einsum('mc,mkc->mk', X, hard_neg_X)
        hard_sim_Y = torch.einsum('mc,mkc->mk', Y, hard_neg_Y)
        
        # Triplet margin loss: max(0, hard_sim - pos_sim + margin)
        # We want: pos_sim > hard_sim (by at least margin)
        triplet_loss_X = F.relu(hard_sim_X - pos_sim.unsqueeze(1) + margin).mean()
        triplet_loss_Y = F.relu(hard_sim_Y - pos_sim.unsqueeze(1) + margin).mean()
        
        hard_loss = (triplet_loss_X + triplet_loss_Y) / 2.0

    # ===== Total loss =====
    loss = pair_loss + hard_neg_weight * hard_loss

    with torch.no_grad():
        conf12 = torch.exp( conf_matrix12 ).max(dim=-1)[0]
        conf21 = torch.exp( conf_matrix21 ).max(dim=-1)[0]
        conf = conf12 * conf21

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

def compute_descriptor_consistency_target(
    f_inv,
    visibility,
    eps=1e-6,
):
    """
    Multi-view descriptor consistency target.

    Args:
        f_inv: [N, V, C]
        visibility: [N, V] bool

    Returns:
        var_target: [N]
    """

    # ===== normalize descriptor =====
    f = F.normalize(f_inv, dim=-1)

    vis = visibility.float().unsqueeze(-1)  # [N,V,1]

    # ===== masked mean =====
    denom = vis.sum(dim=1, keepdim=True).clamp(min=1.0)

    mean_f = (f * vis).sum(dim=1, keepdim=True) / denom

    # ===== squared deviation =====
    sq_dev = ((f - mean_f) ** 2).sum(dim=-1)  # [N,V]

    # ===== mask invisible views =====
    sq_dev = sq_dev * visibility.float()

    # ===== average variance =====
    var_target = sq_dev.sum(dim=1) / (
        visibility.float().sum(dim=1).clamp(min=1.0)
    )
    
    # robust normalization
    p95 = torch.quantile(
    var_target.detach(),
    0.95
    )

    var_target = var_target / (p95 + 1e-6)

    # VERY IMPORTANT
    var_target = torch.clamp(
        var_target,
        0.0,
        1.0
    )
   

    return var_target.detach()

def sigma_consistency_loss(
    f_inv,
    sigma_pred,
    visibility,
):
    """
    Supervise sigma using multi-view descriptor consistency.
    
    Args:
        f_inv: [N,V,C]
        sigma_pred: [N,V,1]
        visibility: [N,V]
    """

    # ===== target =====
    var_target = compute_descriptor_consistency_target(
        f_inv,
        visibility,
    )  # [N]

    # ===== prediction =====
    sigma_pred = sigma_pred.squeeze(-1)  # [N,V]

    # ===== average predicted sigma =====
    sigma_mean = (
        sigma_pred * visibility.float()
    ).sum(dim=1) / (
        visibility.float().sum(dim=1).clamp(min=1.0)
    )

    # ===== robust regression =====
    loss = F.smooth_l1_loss(
        sigma_mean,
        var_target,
    )

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




