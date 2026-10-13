"""Token pruning utilities for MedPruner++ (training-free visual token
compression for 3D medical image understanding).

Terminology follows the MedPruner++ paper (docs/MedPruner++); code
identifiers kept for API stability map to paper terms as follows:
  frame / per-frame        slice s (CT/MRI slice) and per-unit quantities
  frame_group_fusion       ordered support construction (support units I_m)
  uniqueness (grouping)    grouping threshold eps_f
  token_allocation         budget allocation (capacities b_m, cap C_max)
  max_token_per_item       configured capacity limit C_max
  uniqueness (SDC)         relation threshold tau_nbr
  kept/selected token      anchor;  col_mask entries -> neighborhood R_p
  merge / fusion step      local refinement (anchor-neighbor averaging)
  cal_rate                 attention-calibrated capacity (bounded cubic rule)
  cal_compose_rate         realized retention fraction rho
"""
import torch
import torch.nn as nn

def cal_compose_rate(mask):
  """Realized retention fraction rho of a boolean keep mask (kept / total)."""
  return mask.sum()/(mask.shape[0])

def cal_mainfold(metric):
  """Pairwise radial-angular distances d_H in the Lorentz model.

  Native features are mapped to the hyperboloid of curvature -1 by the
  origin exponential map phi(.); the Lorentz distance d_H(x, y) is the
  radial-angular relation: it combines feature magnitude (radial) and
  direction (angular), retaining magnitude differences discarded by cosine
  similarity.

  Args:
    metric: Native feature matrix of shape (N, D) whose rows are compared.

  Returns:
    Distance matrix of shape (N, N) with entries d_H(x_i, x_j); smaller
    means more related (cf. frame_group_fusion: slices are grouped while
    d_H <= eps_f).
  """
  import geoopt
  # --- Radial-angular (Lorentz) distances ---
  num_frames = metric.shape[0]
  D = metric.shape[1]
  k = 1.0
  man = geoopt.Lorentz(k=k)
  scale = 1.0

  # Origin exponential map phi(.): lift D-dim native features onto the hyperboloid of curvature -1
  v = torch.cat(
    [torch.zeros(num_frames, 1, device=metric.device, dtype=metric.dtype),
     scale * metric],
    dim=-1
  )  # [num_frames, D+1]

  # phi(x) = (cosh r, sinh(r)/r * x) with r = ||x||_2
  frames_h = man.expmap0(v)  # [num_frames, D+1]

  # Pairwise Lorentz distances d_H(x, y) (radial-angular relation)
  u = man.dist(
    frames_h.unsqueeze(1),  # [num_frames, 1, D+1]
    frames_h.unsqueeze(0)   # [1, num_frames, D+1]
  )  # [N, N]; smaller d_H means more similar
  return u

def cal_rate(attn, eps=1e-12):
  """Attention-calibrated capacity: mean normalized entropy H_bar of attention
  rows, followed by the bounded cubic rule rho_cal = clip_[0, rho_max](H_bar^3 / 2)."""
  # gemma_attn = tensor(0.5678) -> 0.1
  # hulu_attn = tensor(0.8537) -> 0.3
  # qwen3vl_attn = tensor(0.9238) -> 0.3
  # qwen25vl_attn = tensor(0.9618) -> 0.3

  # Normalize attention rows: a_i = alpha_i / sum_j alpha_j
  attn = attn / attn.sum(dim=-1, keepdim=True)
  # Row-wise normalized entropy; the mean over rows is H_bar
  entropy = -torch.sum(attn * torch.log(attn + eps), dim=-1)
  max_entropy = torch.log(torch.tensor(attn.shape[-1], dtype=attn.dtype))

  normalized_entropy = entropy / max_entropy
  # Bounded cubic rule: rho_cal = clip_[0, rho_max](H_bar^3 / 2), cap rho_max = 0.3
  # rate = normalized_entropy.mean()** 2 / 3 # auto_1
  rate = torch.clamp(normalized_entropy.mean()**3 / 2, min=0, max=0.3)  # auto_2
  return rate

def select_static_windows(feat, batch_size, tau, max_window_size):
  """Legacy dynamic-programming slice-window selection.

  Not part of the MedPruner++ content path; kept for reference.
  """
  dtype = feat.dtype
  metric = nn.functional.normalize(feat, p=2, dim=-1)
  # feature_sim.shape = [31,256]
  feature_sim = 1-torch.nn.functional.cosine_similarity(
    metric[:-1], metric[1:], dim=-1
  )
  # Redundancy-aware grouping of contiguous slices into windows (dynamic programming)
  def get_pruned_static_count_vectorized(feature_sim, batch_size, tau):
    similarity_matrix = torch.ones((batch_size, batch_size, feature_sim.shape[1]), device=feature_sim.device)

    for start in range(batch_size-1):
      cum_similarity = torch.cumprod(feature_sim[start:] > tau, dim=0)
      similarity_matrix[start, start+1:start+1+len(cum_similarity)] = cum_similarity

    window_lengths = torch.arange(batch_size, device=feature_sim.device).unsqueeze(0) - \
                    torch.arange(batch_size, device=feature_sim.device).unsqueeze(1)
    window_lengths = window_lengths.clamp(min=0)

    pruned_static_count = (similarity_matrix.sum(dim=-1) * window_lengths).float()
    return pruned_static_count

  pruned_static_count = get_pruned_static_count_vectorized(feature_sim, batch_size, tau)

  dp = torch.zeros(batch_size, device=pruned_static_count.device)
  prev = torch.zeros(batch_size, dtype=torch.long, device=pruned_static_count.device)
  # [prev[i], i]

  for i in range(batch_size):
    max_val = dp[i-1] if i > 0 else 0
    best_j = i

    for window_size in range(2, min(i+1, max_window_size) + 1):
      j = i - window_size
      current_val = (dp[j] if j>=0 else 0) + pruned_static_count[j+1, i]  # [-, j] + [j+1, i]
      if current_val > max_val:
        max_val = current_val
        best_j = j+1

    dp[i] = max_val
    prev[i] = best_j    # [best_j, i]

  selected_frames = []
  i = batch_size - 1
  while i >= 0:
    selected_frames.append((prev[i].item(), i))
    i = int(prev[i].item()) - 1

  new_feat = []
  for (start,end) in selected_frames:
    new_feat.append(feat[start:end].sum(0))
  selected_frames = selected_frames[::-1]
  new_feat = torch.stack(new_feat).to(dtype)
  selected_ids = torch.tensor(list(range(new_feat.shape[0])))
  return new_feat, selected_ids

def frame_group_fusion(feat, uniqueness = 0.005, metric=None, min_interval = 10000):
  """Ordered support construction: partition contiguous slices into ordered
  support units and aggregate each unit by position-wise averaging.

  Slice descriptors z^f (mean patch feature per slice) are compared with the
  Lorentz distance d_H; consecutive slices with d_H <= grouping threshold
  (`uniqueness`, eps_f in the paper) join the same unit, preserving depth
  order. Averaging stays in the native feature space.
  """
  dtype = feat.dtype
  feat = feat.to(torch.float32)
  metric = feat.mean(1)
  # Slice descriptor z_s^f = mean patch feature, shape [S, d]
  u = cal_mainfold(metric)
  new_feat = []
  num= 1
  selected_ids = [0]
  for i in range(1, feat.shape[0]):
    if u[i,selected_ids[-1]] <= uniqueness and num<=min_interval:
      num+=1
    else:
      new_feat.append(feat[i-num:i].sum(0)/num)
      num = 1
      selected_ids.append(i)

  new_feat.append(feat[feat.shape[0]-num:feat.shape[0]].sum(0)/num)
  feat = torch.stack(new_feat).to(dtype)
  selected_ids = torch.tensor(selected_ids)

  return feat, selected_ids

def token_allocation(feat, max_tokens=10000, max_token_per_item=256, global_metric=None):
  """Budget allocation: per-unit capacities b_m from radial-angular
  allocation logits.

  Pool channels into a patch profile, compute mean relation distances and
  allocation logits eta_m = sqrt(M) * (mean(d_H) - d_H_m) with softmax
  weights pi_m, then apply capped allocation with limit C_max
  (`max_token_per_item`) under the global budget B (`max_tokens`). Residual
  tokens after clipping are redistributed in equal integer batches among
  eligible units.
  """
  if global_metric is not None:
    num_frames = global_metric.shape[0]
    sims = cal_mainfold(global_metric)
  else:
    feat = feat.clone().detach().to(torch.float32)
    num_frames = feat.shape[0]
    global_feat = feat.mean(dim=-1)  # patch profile z^a (channel pooling), shape: (num_slices, num_patches)
    sims = cal_mainfold(global_feat)
    del feat


  # Mean relation distance A_bar_m and allocation logits
  # eta_m = sqrt(M) * (A_bar - A_bar_m) (Eq. allocation_logit)
  sims = sims.mean(dim=-1)
  uniques = sims.mean()-sims
  uniques = uniques * (num_frames ** 0.5)

  max_token_per_item = min(max_token_per_item, max(1, int((max_tokens // num_frames) * 2)))  # c = min(C_max, max(1, 2*floor(B/M)))

  tokens_per_frame = torch.tensor([max_token_per_item] * len(uniques)).to(uniques.device)
  # If M * c > B: allocate b_m = min(c, floor(B * pi_m)), clip at c, and
  # redistribute residual tokens in equal integer batches (Eq. integer_allocation).
  if len(uniques) > max_tokens // max_token_per_item:
    normalized_similarities = torch.softmax(uniques, dim=-1)  # allocation weights pi_m
    tokens_per_frame = torch.floor(normalized_similarities * max_tokens).int()
    mask = tokens_per_frame>max_token_per_item
    tokens_per_frame[mask] = max_token_per_item
    extra_tokens = max_tokens - tokens_per_frame.sum()
    extra_tokens = max(0, extra_tokens)
    pad_num = (~mask).sum()
    avg_pad_tokens = extra_tokens // pad_num
    while mask.sum() > 0 and avg_pad_tokens>0:
      tokens_per_frame[~mask] += avg_pad_tokens
      mask = tokens_per_frame>max_token_per_item
      tokens_per_frame[mask] = max_token_per_item
      extra_tokens = max_tokens - tokens_per_frame.sum()
      extra_tokens = max(0, extra_tokens)
      pad_num = (~mask).sum()
      avg_pad_tokens = extra_tokens // pad_num
  return tokens_per_frame

def spatial_dynamic_compression(
  feats,
  metric=None,
  tokens_per_frame=None,
  uniqueness = 0.2,
  selected_frame_ids=None,
  original_num_frames=None,
  return_token_mask=False,
):
  """
  Spatial Dynamic Compression (SDC): iterative token composition within each
  support unit (slice). Detects redundant patch tokens via the radial-angular
  relation and refines kept anchors by local refinement, reducing the token
  count along the in-plane dimension.

  Note: this variant uses greedy neighbor exclusion, so the retained count
  emerges from the relation threshold. The paper's default composition fixes
  the retained count first (TopK by distinctiveness; see SDC v2 below).

  Args:
    feats: Patch features of shape (num_slices, num_patches, feat_dim).
    metric: Features for the radial-angular relation; defaults to `feats`.
    tokens_per_frame: Per-unit capacity b_m (max anchors kept per slice); no
      limit if None.
    uniqueness: Relation threshold tau_nbr; pairs with 1 - S^H below it are
      neighbors and get merged.
    selected_frame_ids: Indices of the kept slices in the original sequence
      (used to map the token mask).
    original_num_frames: Number of slices in the original sequence (used to
      map the token mask).
    return_token_mask: If True, also return a bool mask marking kept anchors.

  Returns:
    (refined_anchors, token_mask) when return_token_mask=True, otherwise
    refined_anchors (kept per-slice token features).
  """
  all_selected_ids = []  # indices of kept tokens per frame
  new_feats = []
  num_frames, num_tokens, feat_dim = feats.shape

  # Copy the original features; refinement stays in the native feature space
  feats_copy = feats.clone().to(torch.float32)

  # Pairwise Lorentz distances D_pq = d_H(g_p, g_q) within each unit (chunked to bound memory)
  if metric is None:
    metric = feats
  import geoopt
  with torch.no_grad():
    feats_detached = metric.detach().to(torch.float32)
    B, T, D = feats_detached.shape
    man = geoopt.Lorentz(k=1.0)
    scale = 1.0
    sim_all = torch.zeros(B, T, T, device=feats_detached.device, dtype=feats_detached.dtype)
    chunk_size = 256
    for b in range(B):
      flat = feats_detached[b]  # (T, D)
      v = torch.cat(
        [torch.zeros(T, 1, device=flat.device, dtype=flat.dtype),
         scale * flat],
        dim=-1
      )  # (T, D+1)
      pts_h = man.expmap0(v)  # (T, D+1)
      # Chunked distance computation to avoid a (T, T, D+1) OOM
      for i in range(0, T, chunk_size):
        dist_rows = []
        for j in range(0, T, chunk_size):
          d = man.dist(
            pts_h[i:i+chunk_size].unsqueeze(1),
            pts_h[j:j+chunk_size].unsqueeze(0)
          )
          dist_rows.append(d)
        sim_all[b, i:i+chunk_size, :] = -torch.cat(dist_rows, dim=1)
    # Row-wise min-max normalization of -D to similarities S^H in [0, 1]
    sim_all = (sim_all - sim_all.min(dim=-1, keepdim=True)[0]) / \
              (sim_all.max(dim=-1, keepdim=True)[0] - sim_all.min(dim=-1, keepdim=True)[0] + 1e-8)

  # Mean normalized similarity per token; a smaller mean means higher distinctiveness u^H
  all_sim_avg = sim_all.mean(dim=-1)
  # Neighborhood relation: q in R_p iff 1 - S^H_pq < tau_nbr (`uniqueness`)
  col_mask_all = 1 - sim_all < uniqueness

  # Rank tokens by distinctiveness u^H: the most distinctive anchors come first
  unique_sorted_indices = torch.argsort(all_sim_avg, dim=-1, descending=False).to('cpu')

  # --- Per-unit (slice) processing ---
  for num in range(num_frames):
    sorted_indices = unique_sorted_indices[num]
    col_mask = col_mask_all[num]

    device = sorted_indices.device
    N = feats[num].shape[0]  # number of patch tokens in the current unit

    sorted_indices = sorted_indices.to(torch.long)
    # Reorder the neighborhood mask to follow the distinctiveness ranking
    col_mask_tensor = col_mask[sorted_indices][:, sorted_indices]
    col_mask_tensor = col_mask_tensor.to(torch.long)
    all_indices = torch.arange(N, device=device)

    # Upper-triangular mask: only neighbors ranked after the current anchor are considered (greedy exclusion)
    causal_mask = (col_mask_tensor.bool() & torch.triu(torch.ones_like(col_mask_tensor), diagonal=1).bool())
    causal_mask_bool = causal_mask > 0
    mask = torch.ones_like(all_indices, dtype=torch.bool)

    # --- Greedy anchor selection ---
    # Greedy neighbor exclusion: keep a token only if it is not a neighbor of any
    # already-kept anchor; otherwise mark it as merged. Kept anchors absorb the
    # information of their neighbors during refinement. (The paper's default
    # composition replaces this with a fixed-count TopK selection.)
    keep_indices = torch.arange(N, device=device)
    while keep_indices.shape[0] != 0:
      mask[keep_indices] = True
      # Tokens that are neighbors of a kept anchor (to be merged into it)
      remove_indices = torch.where(causal_mask_bool[keep_indices].any(dim=0))[0]
      mask[remove_indices] = False
      # Kept anchors that are themselves covered by other kept anchors
      rest_indices = all_indices[mask]
      rest_indices = torch.where(causal_mask_bool[rest_indices].any(dim=0))[0]
      remove_indices = torch.where(causal_mask_bool[remove_indices].any(dim=0))[0]
      # Keep only tokens covered exclusively by keep_indices, and iterate
      keep_indices = remove_indices[~torch.isin(remove_indices, rest_indices)]

    # Map kept anchors back to their source-patch indices
    all_indices = all_indices[mask]
    select_indices = sorted_indices[all_indices]

    # --- Local refinement: anchor-neighbor averaging, x_tilde = (g + mu) / 2 ---
    mask_valid = col_mask == 1  # neighborhood mask R_p per position
    gathered = feats_copy[num]
    sum_feat = torch.matmul(mask_valid.float(), gathered)  # Neighborhood mean mu via matmul; avoids an (N,N,D) intermediate tensor (OOM)
    counts = mask_valid.sum(1).clamp(min=1).unsqueeze(-1)  # |R_p| with a max(1, .) guard
    fused_all = (sum_feat / counts).to(torch.float32)  # neighborhood mean mu
    # Anchor-neighbor averaging: x_tilde = (g + mu) / 2
    feats[num][select_indices] = ((fused_all[select_indices] + feats_copy[num][select_indices]) / 2).to(feats.dtype)
    select_indices = select_indices.tolist()

    all_selected_ids.append(select_indices)

  # Anchors kept per unit
  selected_nums = torch.tensor([len(x) for x in all_selected_ids], device=feats.device)

  # --- Residual redistribution: move spare capacity in equal batches from under-budget to over-budget units ---
  if tokens_per_frame is not None:
    tokens_per_frame_tmp = tokens_per_frame.clone()
    if sum(selected_nums) > sum(tokens_per_frame):
      mask = selected_nums < tokens_per_frame_tmp  # units still under capacity
      # Residual tokens available for redistribution
      remain_tokens = sum(tokens_per_frame) - sum(tokens_per_frame_tmp)
      remain_tokens += sum(tokens_per_frame_tmp[mask]) - sum(selected_nums[mask])
      if remain_tokens > 0:
        # Redistribute the residual in equal integer batches
        avg_tokens = remain_tokens // (~mask).sum()
        tokens_per_frame_tmp[~mask] += avg_tokens
    tokens_per_frame = tokens_per_frame_tmp
  else:
    tokens_per_frame = selected_nums  # no budget: keep every selected anchor

  # --- Build the global keep mask over the original slice-patch grid (return_token_mask=True) ---
  unicomp_token_mask = None
  frame_ids = None  # bound below when return_token_mask is True
  if return_token_mask:
    if original_num_frames is None:
      original_num_frames = num_frames
    if selected_frame_ids is None:
      selected_frame_ids = torch.arange(num_frames, device=feats.device)
    else:
      selected_frame_ids = selected_frame_ids.to(feats.device)
    frame_ids = selected_frame_ids
    unicomp_token_mask = torch.zeros(original_num_frames * num_tokens, dtype=torch.bool, device=feats.device)

  # --- Cap anchors at the per-unit capacity and assemble the output ---
  for i in range(len(selected_nums)):
    select_indices = all_selected_ids[i]
    # Cap at the per-unit capacity b_m
    select_indices = select_indices[: int(tokens_per_frame[i].item())]
    select_indices.sort()  # restore source-patch order
    new_feats.append(feats[i][select_indices])
    if frame_ids is not None and unicomp_token_mask is not None:
      # Mark anchor positions in the global keep mask
      frame_id = int(frame_ids[i].item())
      token_ids = torch.tensor(select_indices, dtype=torch.long, device=feats.device)
      unicomp_token_mask[frame_id * num_tokens + token_ids] = True

  del feats_copy
  if return_token_mask:
    return torch.cat(new_feats, dim=0), unicomp_token_mask
  else:
    return new_feats, unicomp_token_mask

def spatial_dynamic_compression2(
  feats,
  metric=None,
  tokens_per_frame=None,
  uniqueness = 0.2,
  selected_frame_ids=None,
  original_num_frames=None,
  return_token_mask=False,
):
  """
  Radial-angular token composition (paper default): strictly keep the most
  distinctive tokens per unit according to tokens_per_frame.

  Within each support unit, score token distinctiveness u^H from row-
  normalized Lorentz distances, select the TopK most distinctive anchors
  (fixed retained count, no greedy neighbor exclusion), and refine each
  anchor by anchor-neighbor averaging under the relation threshold tau_nbr.
  Every unit keeps exactly tokens_per_frame anchors.

  Args:
    feats: Patch features of shape (num_slices, num_patches, feat_dim).
    metric: Features for the radial-angular relation; defaults to `feats`.
    tokens_per_frame: Per-unit capacity b_m (anchors kept per slice); keep
      all if None.
    uniqueness: Neighbor-relation threshold tau_nbr in (0, 1] applied to
      1 - S^H (Eq. neighbor_relation).
    selected_frame_ids: Indices of the kept slices in the original sequence
      (used to map the token mask).
    original_num_frames: Number of slices in the original sequence (used to
      map the token mask).
    return_token_mask: If True, also return a bool mask marking kept anchors.

  Returns:
    (refined_anchors, token_mask) when return_token_mask=True, otherwise
    refined_anchors (kept per-slice token features).
  """
  all_selected_ids = []  # indices of kept tokens per frame
  new_feats = []
  num_frames, num_tokens, feat_dim = feats.shape

  # Copy the original features; refinement stays in the native feature space
  feats_copy = feats.clone().to(torch.float32)

  # Pairwise Lorentz distances D_pq = d_H(g_p, g_q) within each unit (chunked to bound memory)
  if metric is None:
    metric = feats
  import geoopt
  with torch.no_grad():
    feats_detached = metric.detach().to(torch.float32)
    B, T, D = feats_detached.shape
    man = geoopt.Lorentz(k=1.0)
    scale = 1.0
    sim_all = torch.zeros(B, T, T, device=feats_detached.device, dtype=feats_detached.dtype)
    chunk_size = 256
    for b in range(B):
      flat = feats_detached[b]  # (T, D)
      v = torch.cat(
        [torch.zeros(T, 1, device=flat.device, dtype=flat.dtype),
         scale * flat],
        dim=-1
      )  # (T, D+1)
      pts_h = man.expmap0(v)  # (T, D+1)
      # Chunked distance computation to avoid a (T, T, D+1) OOM
      for i in range(0, T, chunk_size):
        dist_rows = []
        for j in range(0, T, chunk_size):
          d = man.dist(
            pts_h[i:i+chunk_size].unsqueeze(1),
            pts_h[j:j+chunk_size].unsqueeze(0)
          )
          dist_rows.append(d)
        sim_all[b, i:i+chunk_size, :] = -torch.cat(dist_rows, dim=1)
    # Row-wise min-max normalization of -D to similarities S^H in [0, 1]
    sim_all = (sim_all - sim_all.min(dim=-1, keepdim=True)[0]) / \
              (sim_all.max(dim=-1, keepdim=True)[0] - sim_all.min(dim=-1, keepdim=True)[0] + 1e-8)

  # Mean normalized similarity per token; a smaller mean means higher distinctiveness u^H
  all_sim_avg = sim_all.mean(dim=-1)
  # Neighborhood relation R_p for anchor refinement: 1 - S^H_pq < tau_nbr
  col_mask_all = 1 - sim_all < uniqueness

  # Rank tokens by distinctiveness u^H: the most distinctive anchors come first
  unique_sorted_indices = torch.argsort(all_sim_avg, dim=-1, descending=False).to('cpu')

  # --- Per-unit (slice) processing ---
  for num in range(num_frames):
    sorted_indices = unique_sorted_indices[num]
    col_mask = col_mask_all[num]

    N = feats[num].shape[0]  # number of patch tokens in the current unit

    # Anchors to keep in this unit (K_m)
    budget = int(tokens_per_frame[num].item()) if tokens_per_frame is not None else N
    budget = min(budget, N)

    # TopK by distinctiveness u^H: fixed retained count, no greedy neighbor exclusion
    select_indices = sorted_indices[:budget].to(torch.long)

    # --- Local refinement of each selected anchor ---
    mask_valid = col_mask == 1  # neighborhood mask R_p per position
    gathered = feats_copy[num]
    sum_feat = torch.matmul(mask_valid.float(), gathered)  # Neighborhood mean mu via matmul; avoids an (N,N,D) intermediate tensor (OOM)
    counts = mask_valid.sum(1).clamp(min=1).unsqueeze(-1)  # |R_p| with a max(1, .) guard
    fused_all = (sum_feat / counts).to(torch.float32)  # neighborhood mean mu
    # Anchor-neighbor averaging: x_tilde = (g + mu) / 2
    feats[num][select_indices] = ((fused_all[select_indices] + feats_copy[num][select_indices]) / 2).to(feats.dtype)
    select_indices = select_indices.tolist()

    all_selected_ids.append(select_indices)

  # Anchors kept per unit (exactly tokens_per_frame or N)
  selected_nums = torch.tensor([len(x) for x in all_selected_ids], device=feats.device)
  if tokens_per_frame is None:
    tokens_per_frame = selected_nums

  # --- Build the global keep mask over the original slice-patch grid (return_token_mask=True) ---
  unicomp_token_mask = None
  frame_ids = None  # bound below when return_token_mask is True
  if return_token_mask:
    if original_num_frames is None:
      original_num_frames = num_frames
    if selected_frame_ids is None:
      selected_frame_ids = torch.arange(num_frames, device=feats.device)
    else:
      selected_frame_ids = selected_frame_ids.to(feats.device)
    frame_ids = selected_frame_ids
    unicomp_token_mask = torch.zeros(original_num_frames * num_tokens, dtype=torch.bool, device=feats.device)

  # --- Cap anchors at the per-unit capacity and assemble the output ---
  for i in range(len(selected_nums)):
    select_indices = all_selected_ids[i]
    select_indices.sort()  # restore source-patch order
    new_feats.append(feats[i][select_indices])
    if frame_ids is not None and unicomp_token_mask is not None:
      # Mark anchor positions in the global keep mask
      frame_id = int(frame_ids[i].item())
      token_ids = torch.tensor(select_indices, dtype=torch.long, device=feats.device)
      unicomp_token_mask[frame_id * num_tokens + token_ids] = True

  del feats_copy
  if return_token_mask:
    return torch.cat(new_feats, dim=0), unicomp_token_mask
  else:
    return new_feats, unicomp_token_mask

if __name__ == "__main__":
  import numpy as np
  attn = np.load('tmp/gemma_attn.npy')
  attn = torch.from_numpy(attn)
  attn = attn / attn.sum(dim=-1, keepdim=True)
  print(attn.shape)

  eps = 1e-12
  entropy = -torch.sum(attn * torch.log(attn + eps), dim=-1)
  max_entropy = torch.log(torch.tensor(attn.shape[-1], dtype=attn.dtype))
  normalized_entropy = entropy / max_entropy
  avg_norm_entropy = normalized_entropy.mean()
  result = torch.exp(-avg_norm_entropy * 4)
  print(result)
