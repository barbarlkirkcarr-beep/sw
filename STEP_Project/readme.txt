# STEP: Spatio-Temporal Feature Learning Fusion for Pedestrian Trajectory Prediction

## 简介 (Introduction)
本项目是论文 **"Spatio-Temporal Feature Learning Fusion and Visual Scene Endpoint Prediction for Pedestrian Trajectory Prediction"** 的非官方 PyTorch 复现代码。
该模型包含三个核心模块：
1. **Complementary Attention**: 捕获时间维度上的频繁模式与特殊模式。
2. **Sparse Self-Attention**: 捕获空间维度上的稀疏交互（Zero-Softmax）。
3. **Visual Scene Endpoint Prediction**: 使用 U-Net 结合语义分割图预测行人终点。

## 环境要求 (Requirements)
- Python 3.8+
- PyTorch 1.7+
- NumPy

## 文件结构 (Structure)
- `models/`: 模型核心代码 (Temporal, Spatial, Fusion, UNet)
- `config.py`: 超参数配置
- `main.py`: 训练脚本

## 运行方法 (Usage)
直接运行主程序开始训练：
```bash
python main.py