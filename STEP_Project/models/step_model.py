import torch
import torch.nn as nn
from models.temporal import ComplementaryAttention
from models.spatial import SparseSelfAttention
from models.unet import UNet
from models.fusion import GatedFusion
from config import Config

class STEP(nn.Module):
    def __init__(self):
        super(STEP, self).__init__()
        
        # 1. Trajectory Embedding
        self.embedding = nn.Linear(Config.INPUT_DIM, Config.EMBEDDING_DIM)
        
        # 2. Modules
        self.temporal_net = ComplementaryAttention(
            Config.EMBEDDING_DIM, Config.HIDDEN_DIM, Config.NUM_HEADS, Config.ATTN_THRESHOLD
        )
        self.spatial_net = SparseSelfAttention(
            Config.EMBEDDING_DIM, Config.HIDDEN_DIM
        )
        self.fusion_net = GatedFusion(Config.HIDDEN_DIM)
        
        # 3. Endpoint Prediction (Scene input: Semantic + Heatmap)
        self.unet = UNet(in_channels=Config.NUM_SEMANTIC_CLASSES + 1)
        
        # 4. Final Trajectory Decoder
        # Inputs: [Fused_Feature, Noise(z), Predicted_Endpoint, Last_Pos]
        decoder_input_dim = Config.HIDDEN_DIM + Config.NOISE_DIM + 2 + 2 
        
        self.decoder = nn.Sequential(
            nn.Linear(decoder_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, Config.PRED_LEN * Config.INPUT_DIM)
        )

    def forward(self, obs_traj, scene_input, noise=None):
        """
        obs_traj: [Batch, Obs_Len, 2]
        scene_input: [Batch, C, H, W]
        """
        B, Obs_Len, _ = obs_traj.shape
        
        # --- A. Endpoint Prediction ---
        # 输出概率图 [B, 1, H, W]
        pred_goal_map = self.unet(scene_input)
        
        # Soft-argmax 或 采样 获取终点坐标 (这里简化为取最大概率点坐标用于流程)
        # 实际训练时使用 Ground Truth Endpoint 辅助，测试时使用采样
        # 这里演示: Flatten -> Argmax -> Coordinate
        flat_map = pred_goal_map.view(B, -1)
        max_idx = torch.argmax(flat_map, dim=1)
        goal_x = (max_idx % Config.IMG_SIZE[1]).float() / Config.IMG_SIZE[1]
        goal_y = (max_idx // Config.IMG_SIZE[1]).float() / Config.IMG_SIZE[0]
        pred_goal_coord = torch.stack([goal_x, goal_y], dim=1) # [B, 2] (Normalized)

        # --- B. Feature Learning ---
        # Embedding: [B, T, Emb]
        x_emb = self.embedding(obs_traj)
        
        # 1. Temporal: [B, T, Hidden] -> 取最后一个时刻 [B, Hidden]
        F_t_seq = self.temporal_net(x_emb)
        F_t = F_t_seq[:, -1, :]
        
        # 2. Spatial: 
        # 为了简化 Batch 处理，假设 Batch 内所有人都在同一场景交互
        # 实际论文中是 Graph 结构，这里模拟为 Batch 维度的 Self-Attention
        # 取最后一个观测时刻的位置计算空间交互
        curr_step_emb = x_emb[:, -1, :].unsqueeze(0) # 伪造 [1, B, Emb] 把它当成一组人群
        F_s = self.spatial_net(curr_step_emb).squeeze(0) # [B, Hidden]

        # --- C. Fusion ---
        F_final = self.fusion_net(F_t, F_s)
        
        # --- D. Decoding ---
        if noise is None:
            noise = torch.randn(B, Config.NOISE_DIM).to(x_emb.device)
            
        last_pos = obs_traj[:, -1, :] # [B, 2]
        
        # Concatenate: [Feature, Noise, Goal, Last_Pos] [cite: 311-312]
        decoder_in = torch.cat([F_final, noise, pred_goal_coord, last_pos], dim=1)
        
        pred_traj_rel = self.decoder(decoder_in).view(B, Config.PRED_LEN, 2)
        
        # 输出绝对坐标 (假设预测的是偏移量，或者直接预测绝对坐标，视数据集预处理而定)
        # 这里假设直接预测绝对坐标
        return pred_traj_rel, pred_goal_map