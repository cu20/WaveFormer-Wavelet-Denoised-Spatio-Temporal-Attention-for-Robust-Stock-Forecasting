import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
import os
import pandas as pd
from base_model import DailyBatchSamplerRandom

# 解决 OpenMP 库冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 选择要分析的 pkl 文件
pkl_file = 'data/opensource/csi300_dl_train.pkl'

print(f"正在加载文件: {pkl_file}")
print("=" * 80)

# 加载 pkl 文件
with open(pkl_file, 'rb') as f:
    data_sampler = pickle.load(f)

print(f"数据类型: {type(data_sampler)}")
print(f"数据长度: {len(data_sampler) if hasattr(data_sampler, '__len__') else 'N/A'}")
print("=" * 80)

# 使用 DailyBatchSamplerRandom 获取按日期分组的批次（这样每个批次会包含多个股票）
print("通过 DailyBatchSamplerRandom 获取数据样本（按日期分组）...")
sampler = DailyBatchSamplerRandom(data_sampler, shuffle=False)
data_loader = DataLoader(data_sampler, sampler=sampler, drop_last=False)

# 获取第一个批次（应该包含同一天的所有股票）
batch_data = next(iter(data_loader))
print(f"批次数据原始形状: {batch_data.shape}")

# 按照代码中的处理方式：squeeze 去掉第一个维度
data = torch.squeeze(batch_data, dim=0)
print(f"处理后数据形状: {data.shape}")

# 处理维度：如果只有2维，说明只有1个股票，需要添加维度
if len(data.shape) == 2:
    # 只有1个股票的情况，维度是 (T, F)，需要变成 (1, T, F)
    data = data.unsqueeze(0)
    print(f"调整后数据形状（添加股票维度）: {data.shape}")

print(f"数据维度说明: (N, T, F)")
print(f"  N = {data.shape[0]} (股票数量)")
print(f"  T = {data.shape[1]} (时间步数/回看窗口长度)")
print(f"  F = {data.shape[2]} (特征数 + 市场信息 + 标签)")

print("\n" + "=" * 80)
print("详细分析单个样本:")
print("=" * 80)

# 提取特征和标签（按照 base_model.py 中的方式）
feature = data[:, :, 0:-1]  # 所有时间步，所有特征列（除了最后一列）
label = data[:, -1, -1]     # 最后一个时间步的最后一列（标签）

print("\n【完整数据矩阵 (data)】")
print("-" * 80)
print(f"形状: {data.shape}")
print(f"数据类型: {data.dtype}")
print(f"统计信息:")
print(f"  最小值: {torch.min(data).item():.6f}")
print(f"  最大值: {torch.max(data).item():.6f}")
print(f"  均值: {torch.mean(data).item():.6f}")
print(f"  标准差: {torch.std(data).item():.6f}")

print("\n【特征矩阵 (feature = data[:, :, 0:-1])】")
print("-" * 80)
print(f"形状: {feature.shape}")
print(f"数据类型: {feature.dtype}")
print(f"特征维度: {feature.shape[2]} 个特征")
print(f"  说明: 158 个因子特征 + 63 个市场信息 = 221 个特征")
print(f"统计信息:")
print(f"  最小值: {torch.min(feature).item():.6f}")
print(f"  最大值: {torch.max(feature).item():.6f}")
print(f"  均值: {torch.mean(feature).item():.6f}")
print(f"  标准差: {torch.std(feature).item():.6f}")

# 显示特征矩阵的部分内容
num_stocks_to_show = min(3, feature.shape[0])
num_timesteps_to_show = min(3, feature.shape[1])
num_features_to_show = min(10, feature.shape[2])
print(f"\n特征矩阵预览 (前{num_stocks_to_show}个股票，前{num_timesteps_to_show}个时间步，前{num_features_to_show}个特征):")
print(feature[:num_stocks_to_show, :num_timesteps_to_show, :num_features_to_show].numpy())

# 分析特征的分段（根据 main.py 中的配置）
gate_input_start_index = 158
gate_input_end_index = 221
print(f"\n特征分段说明:")
print(f"  列 0-{gate_input_start_index-1}: 因子特征 ({gate_input_start_index} 个)")
print(f"  列 {gate_input_start_index}-{gate_input_end_index-1}: 市场信息 ({gate_input_end_index - gate_input_start_index} 个)")

print("\n【标签 (label = data[:, -1, -1])】")
print("-" * 80)
print(f"形状: {label.shape}")
print(f"数据类型: {label.dtype}")
print(f"标签数量: {label.shape[0]} (对应 {data.shape[0]} 个股票)")
print(f"统计信息:")
print(f"  最小值: {torch.min(label).item():.6f}")
print(f"  最大值: {torch.max(label).item():.6f}")
print(f"  均值: {torch.mean(label).item():.6f}")
print(f"  标准差: {torch.std(label).item():.6f}")
print(f"  NaN 数量: {torch.sum(torch.isnan(label)).item()}")

# 显示部分标签值
print(f"\n标签值预览 (前20个):")
label_preview = label[:20].numpy()
for i, val in enumerate(label_preview):
    if np.isnan(val):
        print(f"  股票 {i}: NaN")
    else:
        print(f"  股票 {i}: {val:.6f}")

# 分析单个股票的样本
print("\n" + "=" * 80)
print("单个股票样本分析 (以第一个股票为例):")
print("=" * 80)

stock_idx = 0
single_stock_data = data[stock_idx, :, :]  # (T, F)
single_stock_feature = feature[stock_idx, :, :]  # (T, F-1)
single_stock_label = label[stock_idx].item()  # 标量

print(f"\n股票 {stock_idx} 的完整数据:")
print(f"  形状: {single_stock_data.shape}")
print(f"  说明: {single_stock_data.shape[0]} 个时间步 × {single_stock_data.shape[1]} 个特征(含标签)")

print(f"\n股票 {stock_idx} 的特征矩阵:")
print(f"  形状: {single_stock_feature.shape}")
print(f"  说明: {single_stock_feature.shape[0]} 个时间步 × {single_stock_feature.shape[1]} 个特征")
print(f"  特征矩阵内容 (所有时间步，前10个特征):")
print(single_stock_feature[:, :10].numpy())

print(f"\n股票 {stock_idx} 的标签:")
if np.isnan(single_stock_label):
    print(f"  值: NaN")
else:
    print(f"  值: {single_stock_label:.6f}")

print("\n" + "=" * 80)
print("【一个训练样本的定义】")
print("=" * 80)
print("""
一个训练样本 = 一个股票在某个时间点的完整信息

具体结构：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 输入特征 (X): 
   形状: (T, F_feat) = (8, 221)
   - T = 8: 回看窗口长度，包含过去8个时间步的特征
   - F_feat = 221: 特征维度
     * 列 0-157: 158 个因子特征
     * 列 158-220: 63 个市场信息特征
   
   每个时间步的特征向量: (221,)
   8个时间步组成一个序列: (8, 221)

2. 标签 (y):
   形状: 标量 (float)
   - 值: 未来收益率（forward return）
   - 提取位置: 最后一个时间步的最后一列
   - 含义: 基于过去8个时间步的特征，预测的未来收益率

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

示例（股票 0）:
  - 输入 X: 形状 (8, 221) - 8个历史时间步，每个时间步221个特征
  - 标签 y: 0.117791 - 该股票的未来收益率

训练过程:
  - 模型接收 (8, 221) 的特征序列
  - 输出一个预测值（未来收益率）
  - 与真实标签 y 进行比较，计算损失

批次处理:
  - 一个批次包含同一天的所有股票（如292个股票）
  - 批次形状: (N, 8, 221) 其中 N 是当天股票数量
  - 模型可以同时处理多个股票，利用股票间的相关性
""")

print("\n" + "=" * 80)
print("数据格式总结:")
print("=" * 80)
print("""
数据格式: (N, T, F)
  - N: 股票数量（每个批次可能不同，因为按日期分组）
  - T: 时间步数（回看窗口长度，通常为 8）
  - F: 总特征数 = 221 (特征) + 1 (标签) = 222

特征提取:
  - feature = data[:, :, 0:-1]  # 形状: (N, T, 221)
  - label = data[:, -1, -1]     # 形状: (N,)

特征组成:
  - 列 0-157: 158 个因子特征
  - 列 158-220: 63 个市场信息
  - 列 221: 标签（仅在最后一个时间步有效）

标签说明:
  - 标签是未来收益率（forward return）
  - 可能存在 NaN 值（需要处理）
""")

print("\n分析完成！")
