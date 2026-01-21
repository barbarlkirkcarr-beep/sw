import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import zero_softmax

class SparseSelfAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super(SparseSelfAttention, self).__init__()
        self.scale = hidden_dim ** -0.5
        
        self.W_q = nn.Linear(input_dim, hidden_dim)
        self.W_k = nn.Linear(input_dim, hidden_dim)
        self.W_v = nn.Linear(input_dim, hidden_dim)

        # 用于生成稀疏 Mask 的卷积层 (Eq. 12)
        # 输入是一个 N x N 的注意力矩阵，视为单通道图像处理
        self.conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)

    def forward(self, x):
        # x: [Batch, Num_Pedestrians, Dim] (这是单个时间步的输入)
        B, N, D = x.shape
        
        # 1. 线性映射 (Eq. 10)
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        # 2. Dense Attention Matrix (Eq. 11)
        # [B, N, N]
        A_dense = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # 3. 生成 Sparse Mask (Eq. 12)
        # 增加 Channel 维度以进行 2D 卷积: [B, 1, N, N]
        A_input = A_dense.unsqueeze(1)
        # Sigmoid 激活
        A_conv = torch.sigmoid(self.conv(A_input)).squeeze(1)
        
        # 稀疏化操作: ReLU(A - A_conv)
        # 这里实际上是做了一个软阈值处理，不仅置零了部分值，还保留了梯度
        A_sparse = F.relu(A_dense - A_conv)
        
        # 4. Zero-Softmax 归一化 (Eq. 13)
        # 这一步至关重要，防止 0 值在 softmax 后变成非 0
        A_norm = zero_softmax(A_sparse, dim=-1)

        # 5. 聚合特征 (Eq. 14)
        F_s = torch.matmul(A_norm, v)
        
        return F_s # [Batch, Num_Pedestrians, Hidden_Dim]