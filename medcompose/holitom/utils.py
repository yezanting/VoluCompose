import torch
import torch.nn as nn
import math

def cal_compose_rate(mask):
  return mask.sum()/(mask.shape[0])

# image_features.shape = torch.Size([32, 729, 3584])
# attn_weights.shape = torch.Size([32, 729])
# image_feat.shape = torch.Size([32, 729, 3584])
# image_feat.shape = torch.Size([32, 196, 3584])
# attn_weights.shape = torch.Size([32, 196, 1])

def cluster_dpc_knn(x, cluster_num, k=7):
  with torch.no_grad():
    batch_size, seq_len, embed_dim = x.shape
    
    dist_matrix = torch.cdist(x.float(), x.float()) / (embed_dim ** 0.5)    # (batch_size, seq_len, seq_len)
    
    # get local density
    dist_nearest, index_nearest = torch.topk(dist_matrix, k, dim=-1, largest=False) # (batch_size, seq_len, k)
    density = (-(dist_nearest ** 2).mean(dim=-1)).exp() # (batch_size, seq_len)
    # add a little noise to ensure no tokens have the same density.
    density = density + torch.rand(
        density.shape, device=density.device, dtype=density.dtype) * 1e-6
    
    # get distance indicator
    mask = (density[:, None, :] > density[:, :, None]).type(x.dtype)
    dist_max = dist_matrix.flatten(1).max(dim=-1).values[:, None, None]
    dist, index_parent = (dist_matrix * mask + dist_max * (1 - mask)).min(dim=-1)
    
    # select the cluster center according to the score
    score = dist * density
    _, index_center = score.topk(cluster_num, dim=-1)
    
    return index_center, dist_matrix

def select_static_windows(feature_sim, batch_size, tau, max_window_size):
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
    i = prev[i].item() - 1
  
  selected_frames = selected_frames[::-1]
  total_reduced = dp[-1].item()
  
  return selected_frames, total_reduced

def get_static_dynamic_features(image_feat, attn_weights, selected_frames, feature_sim, tau):
  # attn_weights: (batch_size, seq_len)
  batch_size, seq_len, embed_dim = image_feat.shape
  static_feat_list, dynamic_feat_list, _, dynamic_attn_list  = [], [], [], []
  static_pos_list, dynamic_pos_list = [], []
  for start,end in selected_frames:
    all_indices = torch.arange(seq_len, device=image_feat.device).unsqueeze(0)   # (1, seq_len)
    if start == end:
      static_feat_list.append(torch.empty((0, embed_dim), device=image_feat.device))
      # static_attn_list.append(torch.empty((0,), device=attn_weights.device))
      dynamic_feat_list.append(image_feat[start:end+1])
      dynamic_attn_list.append(attn_weights[start:end+1])
      
      static_pos_list.append(torch.empty((0, seq_len), device=image_feat.device, dtype=torch.long))
      dynamic_pos_list.append(all_indices)
    else:
      windows_size = end-start+1
      mask = torch.all(
          feature_sim[start:end, :] > tau, dim=0
      )
      static_feat = image_feat[start:end+1, mask]
      # static_attn = attn_weights[start:end+1, mask]
      dynamic_feat = image_feat[start:end+1, ~mask]
      dynamic_attn = attn_weights[start:end+1, ~mask]
      
      static_feat_list.append(static_feat.mean(dim=0))
      # static_attn_list.append(static_attn.mean(dim=0))
      dynamic_feat_list.append(dynamic_feat)
      dynamic_attn_list.append(dynamic_attn)
      
      static_pos_list.append(all_indices[:, mask].expand(1,-1))
      dynamic_pos_list.append(all_indices[:, ~mask].expand(windows_size,-1))
  
  return static_feat_list, dynamic_feat_list, _, dynamic_attn_list, static_pos_list, dynamic_pos_list

def merge_tokens_by_clustering(feat, target_indices, dist_matrix, cluster_num, Beta):
  batch_size, seq_len, embed_dim = feat.shape
  all_indices = torch.arange(seq_len, device=feat.device)
  all_indices = all_indices.unsqueeze(0).expand(batch_size, -1)  # (batch_size, seq_len)
  non_target_indices = torch.zeros((batch_size, seq_len-cluster_num), dtype=torch.long, device=feat.device)
  for b in range(batch_size):
    non_target_mask = ~torch.isin(all_indices[b], target_indices[b])
    non_target_indices[b] = all_indices[b][non_target_mask]
  # non_target_indices (batch_size, seq_len-cluster_num)
  
  non_target_feat = torch.gather(
    feat,
    dim=1,
    index=non_target_indices.unsqueeze(-1).expand(-1, -1, feat.size(-1))
  )   # (batch_size, seq_len-cluster_num, embed_dim)
  
  dist_matrix = torch.gather(
    dist_matrix, 
    dim=1, 
    index=non_target_indices.unsqueeze(-1).expand(-1, -1, dist_matrix.size(-1))
  )   # (batch_size, seq_len-cluster_num, seq_len)
  dist_matrix = torch.gather(
    dist_matrix, 
    dim=2, 
    index=target_indices.unsqueeze(1).expand(-1, dist_matrix.size(1), -1)
  )   # (batch_size, seq_len-cluster_num, cluster_num)
  
  idx_cluster = torch.argmin(dist_matrix, dim=-1) # (batch_size, seq_len-cluster_num)
  
  cluster_tokens = []
  for b in range(batch_size):
    batch_tokens = []
    for i in range(cluster_num):
      mask = (idx_cluster[b] == i)
      if mask.any():
        cluster_features = non_target_feat[b][mask]
        import os
        if os.environ.get("NO_BETA", "0") == "0":
          # rank0_print("USE_BETA")
          cluster_means = cluster_features.mean(dim=0)
          batch_tokens.append(Beta * feat[b][target_indices[b][i]] + (1 - Beta) * cluster_means)
        else:
          # rank0_print("NO_BETA")
          all_features = torch.cat([feat[b][target_indices[b][i]].unsqueeze(0), cluster_features], dim=0)
          batch_tokens.append(all_features.mean(dim=0))
      else:
        batch_tokens.append(feat[b][target_indices[b][i]])
    cluster_tokens.append(torch.stack(batch_tokens))
  cluster_tokens = torch.stack(cluster_tokens)  # shape: (batch_size, cluster_num, embed_dim)
  
  return cluster_tokens

def merge_tokens_by_attention_density(feat, attn, pos, retain_ratio, D, Beta, K):
  batch_size, seq_len, embed_dim = feat.shape
  dominant_num = round(math.ceil(seq_len * retain_ratio) * (1-D))
  contextual_num = math.ceil(seq_len * retain_ratio) - dominant_num
  
  ## Dominant Visual Tokens
  if dominant_num > 0:
    all_indices = attn.topk(dominant_num, dim=1).indices
    mask = torch.ones_like(feat[:, :, 0], dtype=torch.bool, device=feat.device).scatter_(1, all_indices, False)  # (batch_size, seq_len) False means retained tokens
    # finally, (batch_size, dominant_num, embed_dim) compare with feat
    dominant_tokens = feat.masked_select(~mask.unsqueeze(-1)).view(batch_size, dominant_num, embed_dim)
    dominant_pos = pos.masked_select(~mask).view(batch_size, dominant_num)
  else:
    mask = torch.ones_like(feat[:, :, 0], dtype=torch.bool, device=feat.device)
    dominant_tokens = torch.empty((batch_size, 0, embed_dim), device=feat.device)
    dominant_pos = torch.empty((batch_size, 0), device=feat.device, dtype=torch.long)
  
  ## Contextual Visual Tokens
  if contextual_num > 0:
    ### Filter 
    # feat_filtered: (batch_size, seq_len-dominant_num, embed_dim)
    feat_filtered = feat.masked_select(mask.unsqueeze(-1)).view(batch_size, seq_len - dominant_num, embed_dim) 
    contextual_pos = pos.masked_select(mask.unsqueeze(-1)).view(batch_size, seq_len - dominant_num)
    target_indices, dist_matrix = cluster_dpc_knn(feat_filtered, contextual_num, k=min(K,contextual_num))
    target_indices = torch.sort(target_indices, dim=-1)[0]
    contextual_pos = torch.stack([contextual_pos[b][target_indices[b]] for b in range(batch_size)]) # (batch_size, contextual_num)
    # target_indices (batch_size, contextual_num)
    # dist_matrix (batch_size, seq_len-dominant_num, seq_len-dominant_num)
    # assign tokens to the nearest center
    
    contextual_tokens = merge_tokens_by_clustering(feat_filtered, target_indices, dist_matrix, contextual_num, Beta)
  else:
    contextual_tokens = torch.empty((batch_size, 0, embed_dim), device=feat.device)
    contextual_pos = torch.empty((batch_size, 0), device=feat.device, dtype=torch.long)
  
  image_feat = []
  image_pos = []
  for b in range(batch_size):
    batch_tokens = torch.cat([dominant_tokens[b], contextual_tokens[b]], dim=0)
    batch_pos = torch.cat([dominant_pos[b], contextual_pos[b]], dim=0)
    image_feat.append(batch_tokens)
    image_pos.append(batch_pos)
  image_feat = torch.stack(image_feat)  # shape: (batch_size, dominant_num + contextual_num, embed_dim)
  image_pos = torch.stack(image_pos)
  
  return image_feat, image_pos

def merge_tokens_by_density(feat, pos, retain_ratio, Beta, K):
  batch_size, seq_len, embed_dim = feat.shape
  cluster_num = round(seq_len * retain_ratio)
  if cluster_num > 0:
    target_indices, dist_matrix = cluster_dpc_knn(feat, cluster_num, k=min(K,cluster_num))
    target_indices = torch.sort(target_indices, dim=-1)[0]
    image_pos = torch.stack([pos[b][target_indices[b]] for b in range(batch_size)])
    
    cluster_tokens = merge_tokens_by_clustering(feat, target_indices, dist_matrix, cluster_num, Beta)
    image_feat = cluster_tokens
  else:
    image_feat = torch.empty((batch_size, 0, embed_dim), device=feat.device)
    image_pos = torch.empty((batch_size, 0), device=feat.device, dtype=torch.long)
  
  return image_feat, image_pos

def holitom(static_feat, dynamic_feat, dynamic_attn, static_pos, dynamic_pos, window_size, retain_ratio, D, Beta, K, images_dtype):
  seq_len = static_feat.shape[0] + dynamic_feat.shape[1]
  select_mask = torch.full((window_size, seq_len), False, dtype=torch.bool,device=static_feat.device)
  if window_size==1:
    dynamic_feat, dynamic_pos = merge_tokens_by_attention_density(dynamic_feat, dynamic_attn, dynamic_pos, retain_ratio, D, Beta, K)
    feat = dynamic_feat.flatten(0,1)
    select_mask[0, dynamic_pos] = True
    assert feat.shape[0]==select_mask.sum()
    return feat.to(images_dtype),select_mask

  dynamic_feat, dynamic_pos = merge_tokens_by_attention_density(dynamic_feat, dynamic_attn, dynamic_pos, retain_ratio, D, Beta, K)
  static_feat, static_pos = merge_tokens_by_density(static_feat.unsqueeze(0), static_pos, retain_ratio, Beta, K)

  first_dynamic_feat = dynamic_feat[0:1, :]
  first_dynamic_pos = dynamic_pos[0:1, :]
  first_feat = torch.cat([static_feat, first_dynamic_feat], dim=1)   # (batch_size, first_frame_tokens, embed_dim)
  first_pos = torch.cat([static_pos, first_dynamic_pos], dim=1)
  # Sort tokens by their original positions
  first_pos, first_sorted_indices = torch.sort(first_pos, dim=1)

  other_feat = dynamic_feat[1:, :]
  other_pos = dynamic_pos[1:, :]
  other_pos, other_sorted_indices = torch.sort(other_pos, dim=1)

  first_frame_indices = first_pos[0]
  select_mask[0, first_frame_indices] = True
  for frame_idx, frame_indices in enumerate(other_pos):
    select_mask[frame_idx+1, frame_indices] = True
  feat = torch.cat([first_feat.view(-1,first_feat.shape[-1]), other_feat.view(-1,first_feat.shape[-1])])
  assert feat.shape[0]==select_mask.sum()
  return feat.to(images_dtype),select_mask

__all__ = ["holitom", "get_static_dynamic_features", "select_static_windows"]
