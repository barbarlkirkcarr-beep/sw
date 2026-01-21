import torch

class Config:
    # 硬件配置
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 数据集参数 (ETH-UCY 标准设置)
    OBS_LEN = 8             # 观测长度 (3.2s)
    PRED_LEN = 12           # 预测长度 (4.8s)
    INPUT_DIM = 2           # (x, y)
    
    # 视觉场景参数
    IMG_SIZE = (256, 256)   # 输入图像尺寸
    NUM_SEMANTIC_CLASSES = 6 # 论文: sidewalks, buildings, terrain, trees, roads, undefined [cite: 294]
    
    # 模型超参数 [cite: 353]
    EMBEDDING_DIM = 32      # 线性映射维度
    HIDDEN_DIM = 64         # 内部隐藏层维度
    NUM_HEADS = 8           # Complementary Attention Heads
    NOISE_DIM = 16          # 高斯噪声 z 的维度
    ATTN_THRESHOLD = 0.5    # 互补注意力的掩码阈值 ξ (论文未给具体值，通常取0.5)
    
    # U-Net 参数
    UNET_ENC_CHANS = [32, 64, 128, 256, 512]
    UNET_DEC_CHANS = [512, 256, 128, 64, 32]
    
    # 训练参数 [cite: 354]
    BATCH_SIZE = 32
    NUM_EPOCHS = 300
    LEARNING_RATE = 1e-4    # 0.0001
    LAMBDA_MSE = 1.0        # 平衡 Loss 的系数 (论文公式 19)
    K_STEPS = 20            # TTST 采样次数 (Test-Time Sampling Trick) [cite: 309]