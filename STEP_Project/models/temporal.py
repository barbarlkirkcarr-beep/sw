import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config

class ComplementaryAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_heads=8, threshold=0.5):
        super(ComplementaryAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.threshold = threshold

        # 线性映射 Eq. 1
        self.W_q = nn.Linear(input_dim, hidden_dim)
        self.W_k = nn.Linear(input_dim, hidden_dim)
        self.W_v = nn.Linear(input_dim, hidden_dim)

        # 头部卷积 Eq. 4 (融合多头注意力矩阵)
        self.conv_heads = nn.Conv2d(num_heads, 1, kernel_size=1)
        
        # 门控融合部分的线性层 Eq. 8
        self.linear_f_h = nn.Linear(hidden_dim, hidden_dim)
        self.linear_f_g = nn.Linear(hidden_dim, hidden_dim)
        self.linear_r_h = nn.Linear(hidden_dim, hidden_dim)
        self.linear_r_g = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        # x: [Batch, Time, Dim]
        B, T, D = x.shape
        
        # 1. 计算 Q, K, V 并分头
        q = self.W_q(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # 2. 计算 Attention Score (Eq. 2)
        # scores: [B, Heads, T, T]
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        A_t = F.softmax(scores, dim=-1)

        # 3. 生成 Forward/Reverse Masks (Eq. 4 & Eq. 5)
        # 将 Heads 维度视为 Channel 进行 1x1 卷积
        R_f = torch.sigmoid(self.conv_heads(A_t)) # [B, 1, T, T]
        
        M_f = (R_f <= self.threshold).float()        # Frequent mask
        M_r = ((1 - R_f) <= self.threshold).float()  # Reverse mask

        # 4. 应用 Masks 并重新计算 Attention (Eq. 6)
        A_f = F.softmax(A_t * M_f, dim=-1)
        A_r = F.softmax(A_t * M_r, dim=-1)

        # 5. 加权求和 (Eq. 7)
        T_f = torch.matmul(A_f, v).transpose(1, 2).reshape(B, T, -1)
        T_r = torch.matmul(A_r, v).transpose(1, 2).reshape(B, T, -1)

        # 6. 门控融合 (Eq. 8 & 9)
        # Frequent Mode
        F_f = self.linear_f_h(T_f)
        G_f = torch.sigmoid(self.linear_f_g(T_f))
        
        # Special Mode
        F_r = self.linear_r_h(T_r)
        G_r = torch.sigmoid(self.linear_r_g(T_r))
        
        # Softmax Normalization for Gates
        G_cat = torch.stack([G_f, G_r], dim=-1) # [B, T, D, 2]
        G_cat = F.softmax(G_cat, dim=-1)
        
        # Final Fusion
        F_t = F_f * G_cat[..., 0] + F_r * G_cat[..., 1]
        
        return F_t # [Batch, Time, Hidden_Dim]