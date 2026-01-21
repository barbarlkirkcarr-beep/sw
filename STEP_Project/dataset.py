import os
import torch
import numpy as np
from torch.utils.data import Dataset
from config import Config

class PedestrianDataset(Dataset):
    def __init__(self, data_dir, mode='train'):
        self.mode = mode
        self.obs_len = Config.OBS_LEN
        self.pred_len = Config.PRED_LEN
        self.seq_len = self.obs_len + self.pred_len
        
        # 加载数据
        self.data_sequences = self.load_data(data_dir)
        print(f"[{mode.upper()}] 加载完成: {data_dir}, 共找到 {len(self.data_sequences)} 条轨迹序列。")

    def load_data(self, data_dir):
        all_seqs = []
        if not os.path.exists(data_dir):
            print(f"警告: 路径不存在 {data_dir}")
            return []

        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith(".txt"):
                    file_path = os.path.join(root, file)
                    file_seqs = self.process_file(file_path)
                    all_seqs.extend(file_seqs)
        return all_seqs

    def process_file(self, file_path):
        try:
            # 自动处理科学计数法
            data = np.genfromtxt(file_path, delimiter=None) 
        except Exception as e:
            print(f"读取文件失败 {file_path}: {e}")
            return []

        # 检查数据维度
        if data.ndim == 1 or data.shape[0] < self.seq_len:
            return []

        sequences = []
        ped_ids = np.unique(data[:, 1]) # 第1列是 ID
        
        for pid in ped_ids:
            curr_ped_data = data[data[:, 1] == pid]
            
            # 必须按帧号排序 (第0列)，因为读取时可能乱序
            curr_ped_data = curr_ped_data[curr_ped_data[:, 0].argsort()]
            
            if len(curr_ped_data) < self.seq_len:
                continue
            
            num_seqs = len(curr_ped_data) - self.seq_len + 1
            for i in range(num_seqs):
                # ----------------- 关键修改在这里 -----------------
                # 针对你的 8 列数据：[Frame, ID, X, Z, Y, Vx, Vz, Vy]
                # 我们取 Index 2 (X) 和 Index 4 (Y)
                seq = curr_ped_data[i : i+self.seq_len, [2, 4]] 
                # ------------------------------------------------
                sequences.append(seq)
                
        return sequences

    def __len__(self):
        return len(self.data_sequences)

    def __getitem__(self, idx):
        seq = self.data_sequences[idx]
        seq = torch.from_numpy(seq).float()
        
        obs_traj = seq[:self.obs_len]
        pred_gt = seq[self.obs_len:]
        
        # 场景图 (暂时全0)
        c = Config.NUM_SEMANTIC_CLASSES + 1
        scene_input = torch.zeros(c, Config.IMG_SIZE[0], Config.IMG_SIZE[1])
        
        # 终点热力图 GT
        goal_map_gt = torch.zeros(1, Config.IMG_SIZE[0], Config.IMG_SIZE[1])
        
        # 坐标映射 (简单缩放，假设数据单位是米，大概 0-15m)
        # 你的数据 X=8.45, Y=3.58，这个范围在 0-20 之间，适合用 10-15 的缩放系数
        end_point = pred_gt[-1]
        scale_factor = 10.0 
        
        img_x = int(end_point[0] * scale_factor)
        img_y = int(end_point[1] * scale_factor)
        
        img_x = min(max(img_x, 0), Config.IMG_SIZE[1] - 1)
        img_y = min(max(img_y, 0), Config.IMG_SIZE[0] - 1)
        
        goal_map_gt[0, img_y, img_x] = 1.0
        
        return obs_traj, pred_gt, scene_input, goal_map_gt