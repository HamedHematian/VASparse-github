
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import einops as ein

def process_kv_cache(kv_cache, policy):

    processed_kv_cache = []
    
    for layer_i, (key, value) in enumerate(kv_cache):
        batch_size, head_dim, seq_len, hidden_dim = key.shape
        
        policy_expanded = policy[layer_i].unsqueeze(1).expand(batch_size, head_dim, seq_len)
        
        selected_key = key[policy_expanded == 1].view(batch_size, head_dim, -1, hidden_dim)
        selected_value = value[policy_expanded == 1].view(batch_size, head_dim, -1, hidden_dim)
        
        processed_kv_cache.append((selected_key, selected_value))
    
    return tuple(processed_kv_cache)

def process_kv_cache_shared(kv_cache, policy):

  
    return kv_cache

def attn_postprocess_rank(self_attn_weights, v_token_start, v_token_num, text_token_start, t_token_idx, scale, bias):
    '''
    self_attn_weights: [B, H, L, L]
    '''
    self_attn_weights = self_attn_weights.mean(1) # B, L[Q], L[K]

    t_token_idx = t_token_idx[1] + text_token_start
    relation_vis_text = self_attn_weights[:, t_token_idx, v_token_start: v_token_start+v_token_num] # B, L2, L1

    rank = torch.linalg.matrix_rank(relation_vis_text.float()) # rank
    relation_vis_text = relation_vis_text.mean(1) # B, L1

    s_flag = True # layer needs sparsification or not
    if v_token_num - rank.item() != 0:
        mask = torch.zeros_like(relation_vis_text, dtype=bool)
        _, indices = torch.topk(relation_vis_text, min(int(rank.item() * scale + bias), v_token_num - 1), dim=1)
        mask[0][indices] = 1
    else:
        mask = torch.ones_like(relation_vis_text, dtype=bool)
        s_flag = False

    return mask, s_flag, relation_vis_text

def batch_attn_postprocess_rank(self_attn_weights, v_token_start, v_token_num, text_token_start, t_token_idx, scale, bias):
    '''
    self_attn_weights: [B, H, L, L]
    '''
    self_attn_weights = self_attn_weights.mean(1)  # B, L[Q], L[K]

    t_token_idx = t_token_idx[1] + text_token_start
    relation_vis_text = self_attn_weights[:, t_token_idx, v_token_start: v_token_start + v_token_num]  # B, L2, L1

    rank = torch.linalg.matrix_rank(relation_vis_text.float())  # B 个 batch 的 rank
    print(rank)
    relation_vis_text = relation_vis_text.mean(1)  # B, L1

    s_flag = False  
    mask = torch.zeros_like(relation_vis_text, dtype=bool)

    for i in range(relation_vis_text.size(0)):  # B
        current_rank = int(rank[i].item())  
        if v_token_num - current_rank != 0:
            _, indices = torch.topk(relation_vis_text[i], min(int(v_token_num*0.75), v_token_num - 1), dim=0)
            mask[i][indices] = 1
        else:
            mask[i] = torch.ones_like(relation_vis_text[i], dtype=bool)
            s_flag = False 
    return mask, s_flag, relation_vis_text


def attn_postprocess_rank_vasparse(self_attn_weights, v_token_start, v_token_num, text_token_start, t_token_idx, scale, bias, sparse_ratio=0.75):
    '''
    self_attn_weights: [B, H, L, L]
    '''
    self_attn_weights = self_attn_weights.mean(1) # B, L[Q], L[K]

    t_token_idx = t_token_idx[1] + text_token_start
    relation_vis_text = self_attn_weights[:, :, v_token_start: v_token_start+v_token_num] # B, L2, L1

    rank = torch.linalg.matrix_rank(relation_vis_text.float()) # rank
    relation_vis_text = relation_vis_text.mean(1) # B, L1

    s_flag = True # layer needs sparsification or not
    mask = torch.zeros_like(relation_vis_text, dtype=bool)
    _, indices = torch.topk(relation_vis_text, min(int(v_token_num*sparse_ratio), v_token_num - 1), dim=1)

    for i in range(relation_vis_text.shape[0]):
        mask[i][indices[i]] = 1

    return mask, s_flag, relation_vis_text

def attn_postprocess_rank_vasparse_visual(self_attn_weights, v_token_start, v_token_num, text_token_start, t_token_idx, scale, bias, sparse_ratio=0.75, W=None, P=None, λ=1):
    '''
    self_attn_weights: [B, H, L, L]
    '''
    self_attn_weights = self_attn_weights.mean(1) # B, L[Q], L[K]

    t_token_idx = t_token_idx[1] + text_token_start
    relation_vis_text = self_attn_weights[:, :, v_token_start: v_token_start+v_token_num] # B, L2, L1

    rank = torch.linalg.matrix_rank(relation_vis_text.float()) # rank
    relation_vis_text = relation_vis_text.mean(1) # B, L1

    L1 = relation_vis_text.shape[1]
    if W is None:
        W = torch.full((L1,), 1.0).to(device=relation_vis_text.device) 
    if P is None or len(P)==0:
        P = torch.full((L1,), 1.0).to(device=relation_vis_text.device) 
    else:
        P = P[0].to(device=relation_vis_text.device)

 
    s_flag = True # layer needs sparsification or not
    mask = torch.zeros_like(relation_vis_text, dtype=bool)
    q_k_dot_product = relation_vis_text  

    aggregation_scores = W * (q_k_dot_product ** 2) + λ * P

    _, indices = torch.topk(aggregation_scores, min(int(v_token_num*sparse_ratio), v_token_num - 1), dim=1)

    for i in range(relation_vis_text.shape[0]):
        mask[i][indices[i]] = 1

    return mask, s_flag, relation_vis_text

def visual_aware_attn_postprocess_rank(self_attn_weights, v_token_start, v_token_num, text_token_start, t_token_idx, scale, bias, W=None, P=None, λ=0):

    self_attn_weights = self_attn_weights.mean(1)  # B, L[Q], L[K]

    t_token_idx = t_token_idx[1] + text_token_start
    relation_vis_text = self_attn_weights[:, t_token_idx, v_token_start: v_token_start + v_token_num]  # B, L2, L1

    rank = torch.linalg.matrix_rank(relation_vis_text.float())  # rank
    relation_vis_text = relation_vis_text.mean(1)  # B, L1

    L1 = relation_vis_text.shape[1]

    if W is None:
        W = torch.full((L1,), 1.0)  
    if P is None:
        P = torch.full((L1,), 1.0) 

    s_flag = True  

    if v_token_num - rank.item() != 0:
        mask = torch.zeros_like(relation_vis_text, dtype=bool)
        
        q_k_dot_product = relation_vis_text  

        aggregation_scores = W * (q_k_dot_product ** 2) + λ * P

        _, indices = torch.topk(aggregation_scores, min(int(rank.item() * scale + bias), v_token_num - 1), dim=1)

        mask[0][indices] = 1
    else:
        mask = torch.ones_like(relation_vis_text, dtype=bool)
        s_flag = False

    return mask, s_flag, relation_vis_text


def batch_index_select(x, idx):
    if len(x.size()) == 4:
        B, H, N, C = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B*N, H, C)[idx.reshape(-1)].reshape(B, H, N_new, C)
        return out
    elif len(x.size()) == 3:
        B, N, C = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B*N, C)[idx.reshape(-1)].reshape(B, N_new, C)
        return out
    elif len(x.size()) == 2:
        B, N = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B*N)[idx.reshape(-1)].reshape(B, N_new)
        return out
    else:
        raise NotImplementedError




def index_points(points, idx):
    """Sample features following the index.
    Returns:
        new_points:, indexed points data, [B, S, C]

    Args:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def cluster_and_merge(x, cluster_num):
    B, N, C = x.shape

    x1 = ein.rearrange(x, "b l r -> b l () r")
    x2 = ein.rearrange(x, "b l r -> b () l r")
    distance = (x1 - x2).norm(dim=-1, p=2)
    dist_matrix = distance / (C ** 0.5)
    dist_nearest, index_nearest = torch.topk(dist_matrix, k=cluster_num, dim=-1, largest=False)
    density = (-(dist_nearest ** 2).mean(dim=-1)).exp()
    density = density + torch.rand(
        density.shape, device=density.device, dtype=density.dtype) * 1e-6

    mask = density[:, None, :] > density[:, :, None]
    mask = mask.type(x.dtype)
    dist_max = dist_matrix.flatten(1).max(dim=-1)[0][:, None, None]
    dist, index_parent = (dist_matrix * mask + dist_max * (1 - mask)).min(dim=-1)

    score = dist * density
    _, index_down = torch.topk(score, k=cluster_num, dim=-1)

    dist_matrix = index_points(dist_matrix, index_down)

    idx_cluster = dist_matrix.argmin(dim=1)

    idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, cluster_num)
    idx_tmp = torch.arange(cluster_num, device=x.device)[None, :].expand(B, cluster_num)
    idx_cluster[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)

    B, N, C = x.shape
    device = dist_matrix.device
    idx_token = torch.arange(N)[None, :].repeat(B, 1).to(device)
    agg_weight = x.new_ones(B, N, 1)
    
    token_weight = x.new_ones(B, N, 1)
    
    idx_batch = torch.arange(B, device=x.device)[:, None]
    idx = idx_cluster + idx_batch * cluster_num

    all_weight = token_weight.new_zeros(B * cluster_num, 1)
    all_weight.index_add_(dim=0, index=idx.reshape(B * N),
                            source=token_weight.reshape(B * N, 1))      
    all_weight = all_weight + 1e-6

    x_merged = x.new_zeros(B * cluster_num, C)
    source = x * norm_weight
    x_merged.index_add_(dim=0, index=idx.reshape(B * N),
                        source=source.reshape(B * N, C).type(x.dtype))
    x_merged = x_merged.reshape(B, cluster_num, C)
    
    return x_merged
