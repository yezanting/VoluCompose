import torch
import torch.nn as nn
import math

def cal_compose_rate(mask):
  return mask.sum()/(mask.shape[0])

def save_as_np(feat):
  import os
  import numpy as np
  # Save image_feat as npy file
  save_dir = "/root/epfs/zantingye-model_with_code/llm/workspace/feat"
  os.makedirs(save_dir, exist_ok=True)
  import time
  save_path = os.path.join(save_dir, f"image_feat.npy")
  np.save(save_path, feat.detach().cpu().float().numpy())


def frame_group_fusion(feat, uniqueness = 0.005, metric=None, min_interval = 10000):
  dtype = feat.dtype
  feat = feat.to(torch.float32)
  metric = feat.mean(1)
  metric = nn.functional.normalize(metric, dim=-1)
  u = 1-torch.cosine_similarity(metric[:, None, :], metric[None, :, :], dim=-1)
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
  if global_metric is not None:
    num_frames = global_metric.shape[0]
    global_metric = torch.nn.functional.normalize(global_metric, p=2, dim=-1)
    sims = torch.cosine_similarity(global_metric[:, None, :], global_metric[None, :, :], dim=-1)
  else:
    feat = feat.clone().detach().to(torch.float32)
    num_frames = feat.shape[0]
    global_feat = feat.mean(dim=-1)
    global_feat = torch.nn.functional.normalize(global_feat, p=2, dim=-1)
    sims = torch.cosine_similarity(global_feat[:, None, :], global_feat[None, :, :], dim=-1)
    del feat


  sims = sims.mean(dim=-1)
  uniques = sims.mean()-sims
  uniques = uniques * (num_frames ** 0.5)

  max_token_per_item = min(max_token_per_item, max(1, int((max_tokens // num_frames) * 2))) 

  tokens_per_frame = torch.tensor([max_token_per_item] * len(uniques)).to(uniques.device)
  if len(uniques) > max_tokens // max_token_per_item:
    normalized_similarities = torch.softmax(uniques, dim=-1)
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

def token_allocation2(feat, max_tokens=10000, max_token_per_item=256, global_metric=None):
  # save_as_np(feat)
  if global_metric is not None:
    num_frames = global_metric.shape[0]
    global_metric = torch.nn.functional.normalize(global_metric, p=2, dim=-1)
    sims = torch.cosine_similarity(global_metric[:, None, :], global_metric[None, :, :], dim=-1)
  else:
    feat = feat.clone().detach().to(torch.float32)
    num_frames = feat.shape[0]
    global_feat = feat.mean(dim=-1)
    global_feat = torch.nn.functional.normalize(global_feat, p=2, dim=-1)
    sims = torch.cosine_similarity(global_feat[:, None, :], global_feat[None, :, :], dim=-1)
    del feat

  sims = sims.mean(dim=-1)
  uniques = sims * (num_frames ** 0.5)

  max_token_per_item = min(max_token_per_item, max(1, int((max_tokens // num_frames) * 2))) 

  tokens_per_frame = torch.tensor([max_token_per_item] * len(uniques)).to(uniques.device)
  if len(uniques) > max_tokens // max_token_per_item:
    normalized_similarities = torch.softmax(uniques, dim=-1)
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
  nextline_token=torch.tensor([198]),
  selected_frame_ids=None,
  original_num_frames=None,
  return_token_mask=False,
):
  all_selected_ids = []
  new_feats = []
  num_frames, num_tokens, feat_dim = feats.shape

  feats_copy = feats.clone().to(torch.float32)

  if metric is None:
    metric = feats
  feats_norm = torch.nn.functional.normalize(metric.detach(), p=2, dim=-1)
  sim_all = torch.matmul(feats_norm, feats_norm.transpose(1, 2))

  all_sim_avg = sim_all.mean(dim=-1)
  col_mask_all = 1 - sim_all < uniqueness

  unique_sorted_indices = torch.argsort(all_sim_avg, dim=-1, descending=False).to('cpu')

  for num in range(num_frames):
    sorted_indices = unique_sorted_indices[num]
    col_mask = col_mask_all[num]

    device = sorted_indices.device
    N = feats[num].shape[0]

    sorted_indices = sorted_indices.to(torch.long)
    col_mask_tensor = col_mask[sorted_indices][:, sorted_indices]
    col_mask_tensor = col_mask_tensor.to(torch.long)
    all_indices = torch.arange(N, device=device)

    causal_mask = (col_mask_tensor.bool() & torch.triu(torch.ones_like(col_mask_tensor), diagonal=1).bool())
    causal_mask_bool = causal_mask > 0
    mask = torch.ones_like(all_indices, dtype=torch.bool)

    keep_indices = torch.arange(N, device=device)
    while keep_indices.shape[0] != 0:
      mask[keep_indices] = True
      remove_indices = torch.where(causal_mask_bool[keep_indices].any(dim=0))[0]
      mask[remove_indices] = False
      rest_indices = all_indices[mask]
      rest_indices = torch.where(causal_mask_bool[rest_indices].any(dim=0))[0]
      remove_indices = torch.where(causal_mask_bool[remove_indices].any(dim=0))[0]
      keep_indices = remove_indices[~torch.isin(remove_indices, rest_indices)]

    all_indices = all_indices[mask]
    select_indices = sorted_indices[all_indices]

    mask_valid = col_mask == 1
    gathered = feats_copy[num]
    sum_feat = torch.matmul(mask_valid.float(), gathered)
    counts = mask_valid.sum(1).clamp(min=1).unsqueeze(-1)
    fused_all = (sum_feat / counts).to(torch.float32)
    feats[num][select_indices] = ((fused_all[select_indices] + feats_copy[num][select_indices]) / 2).to(feats.dtype)
    select_indices = select_indices.tolist()

    all_selected_ids.append(select_indices)

  selected_nums = torch.tensor([len(x) for x in all_selected_ids], device=feats.device)

  if tokens_per_frame is not None:
    tokens_per_frame_tmp = tokens_per_frame.clone()
    if sum(selected_nums) > sum(tokens_per_frame):
      mask = selected_nums < tokens_per_frame_tmp
      remain_tokens = sum(tokens_per_frame) - sum(tokens_per_frame_tmp)
      remain_tokens += sum(tokens_per_frame_tmp[mask]) - sum(selected_nums[mask])
      if remain_tokens > 0:
        avg_tokens = remain_tokens // (~mask).sum()
        tokens_per_frame_tmp[~mask] += avg_tokens
    tokens_per_frame = tokens_per_frame_tmp
  else:
    tokens_per_frame = selected_nums

  if return_token_mask:
    if original_num_frames is None:
      original_num_frames = num_frames
    if selected_frame_ids is None:
      selected_frame_ids = torch.arange(num_frames, device=feats.device)
    else:
      selected_frame_ids = selected_frame_ids.to(feats.device)
    unicomp_token_mask = torch.zeros(original_num_frames * num_tokens, dtype=torch.bool, device=feats.device)

  for i in range(len(selected_nums)):
    select_indices = all_selected_ids[i]
    select_indices = select_indices[: int(tokens_per_frame[i].item())]
    select_indices.sort()
    new_feats.append(feats[i][select_indices])
    if return_token_mask:
      frame_id = int(selected_frame_ids[i].item())
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
  nextline_token=torch.tensor([198]),
  selected_frame_ids=None,
  original_num_frames=None,
  return_token_mask=False,
):
  all_selected_ids = []
  new_feats = []
  num_frames, num_tokens, feat_dim = feats.shape

  feats_copy = feats.clone().to(torch.float32)

  if metric is None:
    metric = feats
  feats_norm = torch.nn.functional.normalize(metric.detach(), p=2, dim=-1)
  sim_all = torch.matmul(feats_norm, feats_norm.transpose(1, 2))

  all_sim_avg = sim_all.mean(dim=-1)
  col_mask_all = 1 - sim_all < uniqueness

  unique_sorted_indices = torch.argsort(all_sim_avg, dim=-1, descending=False).to('cpu')

  for num in range(num_frames):
    sorted_indices = unique_sorted_indices[num]
    col_mask = col_mask_all[num]

    N = feats[num].shape[0]

    budget = int(tokens_per_frame[num].item()) if tokens_per_frame is not None else N
    budget = min(budget, N)

    select_indices = sorted_indices[:budget].to(torch.long)

    mask_valid = col_mask == 1
    gathered = feats_copy[num]
    sum_feat = torch.matmul(mask_valid.float(), gathered)
    counts = mask_valid.sum(1).clamp(min=1).unsqueeze(-1)
    fused_all = (sum_feat / counts).to(torch.float32)
    feats[num][select_indices] = ((fused_all[select_indices] + feats_copy[num][select_indices]) / 2).to(feats.dtype)
    select_indices = select_indices.tolist()

    all_selected_ids.append(select_indices)

  selected_nums = torch.tensor([len(x) for x in all_selected_ids], device=feats.device)
  if tokens_per_frame is None:
    tokens_per_frame = selected_nums

  if return_token_mask:
    if original_num_frames is None:
      original_num_frames = num_frames
    if selected_frame_ids is None:
      selected_frame_ids = torch.arange(num_frames, device=feats.device)
    else:
      selected_frame_ids = selected_frame_ids.to(feats.device)
    unicomp_token_mask = torch.zeros(original_num_frames * num_tokens, dtype=torch.bool, device=feats.device)

  for i in range(len(selected_nums)):
    select_indices = all_selected_ids[i]
    select_indices.sort()
    new_feats.append(feats[i][select_indices])
    if return_token_mask:
      frame_id = int(selected_frame_ids[i].item())
      token_ids = torch.tensor(select_indices, dtype=torch.long, device=feats.device)
      unicomp_token_mask[frame_id * num_tokens + token_ids] = True

  del feats_copy
  if return_token_mask:
    return torch.cat(new_feats, dim=0), unicomp_token_mask
  else:
    return new_feats, unicomp_token_mask