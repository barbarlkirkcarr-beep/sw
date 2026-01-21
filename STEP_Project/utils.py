import torch
import torch.nn.functional as F
import numpy as np

def zero_softmax(x, dim=-1, epsilon=1e-10):
    """
    论文公式 (13): Zero-softmax function
    Logic: (exp(x) - 1) / (sum(exp(x) - 1) + epsilon)
    用于防止稀疏后的 0 权重在 softmax 后变成非 0 值。
    """
    # 确保输入 x 经过 ReLU 或类似处理非负，或者在这里处理
    # 论文中 A_sparse 是通过 ReLU 得到的，所以是非负的
    numerator = torch.exp(x) - 1
    # 加上 epsilon 防止除零
    denominator = torch.sum(numerator, dim=dim, keepdim=True) + epsilon
    return numerator / denominator

def ade_fde(pred_traj, target_traj):
    """
    计算 Average Displacement Error (ADE) 和 Final Displacement Error (FDE)
    pred_traj: [K, Batch, Pred_Len, 2] - K 为采样次数
    target_traj: [Batch, Pred_Len, 2]
    """
    K, B, L, _ = pred_traj.shape
    
    # 计算所有 K 个采样的误差
    # diff: [K, B, L, 2]
    diff = pred_traj - target_traj.unsqueeze(0)
    diff_sq = diff ** 2
    dist = torch.sqrt(torch.sum(diff_sq, dim=3)) # [K, B, L]
    
    # ADE: 平均每个时间步的距离
    ade_per_sample = torch.mean(dist, dim=2) # [K, B]
    # FDE: 最后一个时间步的距离
    fde_per_sample = dist[:, :, -1] # [K, B]
    
    # 选取 K 个采样中 ADE 最小的那个作为该样本的误差 (Best-of-K) [cite: 344]
    min_ade, _ = torch.min(ade_per_sample, dim=0) # [B]
    min_fde, _ = torch.min(fde_per_sample, dim=0) # [B]
    
    return torch.mean(min_ade).item(), torch.mean(min_fde).item()