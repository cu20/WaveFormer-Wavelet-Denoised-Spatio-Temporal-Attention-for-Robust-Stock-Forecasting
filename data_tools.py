import numpy as np
import pandas as pd
import random
import os

# 特征数量：
# 当前通过 data_pocess.py 构造的数据中，
# 输入特征列为 short_att（不含 label_1d/label_5d/label_20d），共 35 个特征
features_num = 35
model_num = 50

#计算相关系数
def calc_ic(pred, label):
    df = pd.DataFrame({'pred':pred, 'label':label})
    ic = df['pred'].corr(df['label'])
    ric = df['pred'].corr(df['label'], method='spearman')
    return ic, ric

#Z-score标准化
def zscore(x):
    return (x - np.mean(x)) / np.std(x)

#删除极端值
def drop_extreme(x, percent=0.025):
    sorted_indices = tf.argsort(x).numpy()
    n = x.shape[0]
    
    lower_bound = int(n * percent)
    upper_bound = int(n * (1 - percent))
    
    mask = np.zeros(n, dtype=bool)
    mask[sorted_indices[lower_bound:upper_bound]] = True
    
    filtered_x = x[mask]
    return mask, filtered_x

#删除NaN值
def drop_na(x):
    mask = ~np.isnan(x)
    filtered_x = x[mask]
    return mask, filtered_x


def compute_feature_stats(data, features_num=35):
    """
    从训练集计算每个特征的 median 和 MAD（中位数绝对偏差），用于 RobustZScore 标准化。
    验证/测试集必须使用训练集统计量，避免数据泄露。
    返回: dict {'median': (F,), 'mad': (F,)}
    """
    def _to_numpy(x):
        return x.numpy() if hasattr(x, "numpy") else np.asarray(x)

    if data is None or len(data) == 0:
        return None

    feats = []
    for i in range(len(data)):
        f = _to_numpy(data[i][0])
        if f.ndim == 2:
            f = f[:, :features_num]
        feats.append(f)
    stacked = np.concatenate(feats, axis=0)  # (N_total, F)
    stacked = np.nan_to_num(stacked, nan=0.0, posinf=0.0, neginf=0.0)

    median = np.median(stacked, axis=0)
    mad = np.median(np.abs(stacked - median), axis=0)
    mad = np.maximum(mad, 1e-6)  # 避免除零

    return {"median": median.astype(np.float32), "mad": mad.astype(np.float32)}


def robust_zscore_normalize(feat, stats, clip_range=(-3.0, 3.0)):
    """对特征做 RobustZScore 并 clip 到 [-3, 3]。"""
    x = (feat - stats["median"]) / stats["mad"]
    return np.clip(x, clip_range[0], clip_range[1]).astype(np.float32)


# 创建用于训练的数据库(TensorFlow数据集)
# data是包含特征和收益率的元组列表，shuffle为是否打乱数据，batch_size为批量大小
# feature_stats: 由 compute_feature_stats(train_data) 得到，验证/测试集需传入以使用训练集统计量
# def create_tf_dataset(data, shuffle=False, batch_size=400, feature_stats=None):

#     def _to_numpy(x):
#         return x.numpy() if hasattr(x, "numpy") else np.asarray(x)

#     if data is None or len(data) == 0:
#         return []

#     # 统一时间步长：以第一个样本为基准，对其他样本进行截断/补零
#     seq_len = int(_to_numpy(data[0][0]).shape[0])
#     if seq_len <= 0:
#         return []

#     # 创建一个形状为(数据长度, 时间步长, 特征数量+1)的数组，其中最后一个维度是收益率
#     data_arr = np.zeros(shape=(len(data), seq_len, features_num + 1), dtype=np.float32)
#     final_batches = []

#     # 遍历每个数据样本
#     for i in range(len(data)):
#         feat = _to_numpy(data[i][0])
#         ret = _to_numpy(data[i][1])

#         # 期望 feat 为 (T, F)，只取前 features_num 列特征
#         if feat.ndim != 2:
#             raise ValueError(f"features should be 2D (T,F), got shape={feat.shape}")
#         feat = feat[:, :features_num].astype(np.float32)
#         feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)

#         # RobustZScore 特征标准化（MASTER 风格）
#         if feature_stats is not None:
#             feat = robust_zscore_normalize(feat, feature_stats)

#         # 期望 ret 为 (T,)
#         ret = ret.reshape(-1)

#         curr_len = min(seq_len, feat.shape[0], ret.shape[0])
#         if curr_len > 0:
#             data_arr[i, :curr_len, 0:features_num] = feat[:curr_len, :]
#             # 存储所有时间步的标签，后续可以选择使用哪个时间步
#             data_arr[i, :curr_len, features_num] = ret[:curr_len]

#     if shuffle == True:
#         # 如果需要打乱数据，则打乱data_arr
#         np.random.shuffle(data_arr)
#         # 创建一个形状为(数据长度)的数组来存储收益率
#         returns_arr  = np.zeros(shape=(len(data)))
#         # 遍历每个数据样本
#         for i in range(0, len(data)):
#             # 将收益率复制到returns_arr中
#             returns_arr[i] = data_arr[i, seq_len - 1, features_num]
#         # 删除收益率列
#         data_arr = data_arr[:,:,0:features_num]

#         # 准备数据切片
#         stocks_left = data_arr.shape[0] - batch_size
#         i =0 
#         while (stocks_left >= batch_size):
#             # 获取当前批次的特征
#             curr_feature = data_arr[i*batch_size: batch_size + i*batch_size,:,0:features_num]
#             # 获取当前批次的收益率
#             curr_returns = returns_arr[i*batch_size: batch_size + i*batch_size]
#             # 将当前批次添加到最终批次列表中
#             final_batches.append((tf.convert_to_tensor(curr_feature), tf.convert_to_tensor(curr_returns)))
#             # 增加索引
#             i+=1 
#             # 减少剩余股票数量
#             stocks_left = stocks_left - batch_size
        
#         if stocks_left < batch_size and stocks_left > 5:
#             # 获取当前批次的长度
#             curr_len = stocks_left
#             # 获取最后一个条目的索引，最后一个条目是当前批次的长度减去剩余股票数量
#             last_entry =i*batch_size +batch_size
#             # 获取当前批次的长度
#             curr_feature = data_arr[last_entry :last_entry+curr_len,:,0:features_num]
#             # 获取当前批次的长度
#             curr_returns = returns_arr[last_entry :last_entry+curr_len]
#             # 将当前批次添加到最终批次列表中
#             curr_returns = returns_arr[last_entry :last_entry+curr_len]
#             final_batches.append((tf.convert_to_tensor(curr_feature), tf.convert_to_tensor(curr_returns)))
#         # 返回最终批次列表
#         return final_batches
#     # 如果不需要打乱数据，则直接返回数据
#     else:
#         # 创建一个形状为(数据长度)的数组来存储收益率
#         returns_arr  = np.zeros(shape=(len(data)))
#         # 遍历每个数据样本
#         for i in range(0, len(data)):
#             # 将收益率复制到returns_arr中
#             # 选择使用哪个时间步的标签：'first' 或 'last'，默认 'first'
#             label_time_step = os.environ.get('LABEL_TIME_STEP', 'first').lower()
#             if label_time_step == 'first':
#                 returns_arr[i] = data_arr[i, 0, features_num]  # 使用第一个时间步的标签
#             else:
#                 returns_arr[i] = data_arr[i, seq_len - 1, features_num]  # 使用最后一个时间步的标签（原做法）
#         data_arr = data_arr[:,:,0:features_num]
#         stocks_left = data_arr.shape[0] - batch_size
#         i = 0
#         # 准备数据切片
#         while stocks_left >= batch_size:
#             # 获取当前批次的长度
#             curr_feature = data_arr[i*batch_size: batch_size + i*batch_size,:,0:features_num]
#             curr_returns = returns_arr[i*batch_size: batch_size + i*batch_size]
#             final_batches.append((tf.convert_to_tensor(curr_feature), tf.convert_to_tensor(curr_returns)))
#             i+=1 
#             stocks_left = stocks_left - batch_size
        
#         if stocks_left < batch_size and stocks_left > 5:
#             curr_len = stocks_left
#             last_entry =i*batch_size +batch_size
#             curr_feature = data_arr[last_entry :last_entry+curr_len,:,0:features_num]
#             curr_returns = returns_arr[last_entry :last_entry+curr_len]
#             final_batches.append((tf.convert_to_tensor(curr_feature), tf.convert_to_tensor(curr_returns)))
        
#         return final_batches


class DailyBatchSamplerRandom:
    def __init__(self):
        pass


# ============================================================================
# PyTorch Dataset 支持
# ============================================================================

class WaveFormerDataset(Dataset):
    """PyTorch Dataset for WaveFormer model"""
    def __init__(self, features, labels):
        """
        Parameters:
        -----------
        features : np.ndarray or torch.Tensor
            特征数据，形状为 (N, T, F)
        labels : np.ndarray or torch.Tensor
            标签数据，形状为 (N,)
        """
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch is not available. Please install: pip install torch")
        
        if isinstance(features, np.ndarray):
            self.features = torch.from_numpy(features).float()
        else:
            self.features = features.float()
        
        if isinstance(labels, np.ndarray):
            self.labels = torch.from_numpy(labels).float()
        else:
            self.labels = labels.float()
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def create_pytorch_dataset(data, shuffle=False, batch_size=400, feature_stats=None):
    """
    创建用于训练的PyTorch数据集
    
    Parameters:
    -----------
    data : list
        包含特征和收益率的元组列表，每个元素为 (features, labels)
        features: (T, F) 形状的数组
        labels: (T,) 形状的数组
    shuffle : bool
        是否打乱数据
    batch_size : int
        批量大小
    feature_stats : dict
        特征统计量，由 compute_feature_stats(train_data) 得到
        格式: {'median': (F,), 'mad': (F,)}
    
    Returns:
    --------
    DataLoader
        PyTorch DataLoader 对象
    """
    if not PYTORCH_AVAILABLE:
        raise ImportError("PyTorch is not available. Please install: pip install torch")
    
    def _to_numpy(x):
        """将各种类型转换为numpy数组"""
        if hasattr(x, 'numpy'):
            return x.numpy()
        elif hasattr(x, 'cpu'):
            return x.cpu().numpy()
        else:
            return np.asarray(x)

    if data is None or len(data) == 0:
        return None

    # 统一时间步长：以第一个样本为基准
    seq_len = int(_to_numpy(data[0][0]).shape[0])
    if seq_len <= 0:
        return None

    # 创建数组存储特征和标签
    data_arr = np.zeros(shape=(len(data), seq_len, features_num + 1), dtype=np.float32)
    
    # 遍历每个数据样本
    for i in range(len(data)):
        feat = _to_numpy(data[i][0])
        ret = _to_numpy(data[i][1])

        # 期望 feat 为 (T, F)，只取前 features_num 列特征
        if feat.ndim != 2:
            raise ValueError(f"features should be 2D (T,F), got shape={feat.shape}")
        feat = feat[:, :features_num].astype(np.float32)
        feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)

        # RobustZScore 特征标准化
        if feature_stats is not None:
            feat = robust_zscore_normalize(feat, feature_stats)

        # 期望 ret 为 (T,)
        ret = ret.reshape(-1)

        curr_len = min(seq_len, feat.shape[0], ret.shape[0])
        if curr_len > 0:
            data_arr[i, :curr_len, 0:features_num] = feat[:curr_len, :]
            data_arr[i, :curr_len, features_num] = ret[:curr_len]

    # 提取特征和标签
    features = data_arr[:, :, 0:features_num]  # (N, T, F)
    
    # 选择使用哪个时间步的标签
    label_time_step = os.environ.get('LABEL_TIME_STEP', 'first').lower()
    if label_time_step == 'first':
        labels = data_arr[:, 0, features_num]  # 使用第一个时间步的标签
    else:
        labels = data_arr[:, seq_len - 1, features_num]  # 使用最后一个时间步的标签
    
    # 如果需要打乱数据
    if shuffle:
        indices = np.arange(len(features))
        np.random.shuffle(indices)
        features = features[indices]
        labels = labels[indices]
    
    # 创建 PyTorch Dataset
    dataset = WaveFormerDataset(features, labels)
    
    # 创建 DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,  # 保留最后一个不完整的批次
        num_workers=0,  # Windows上建议设为0，避免多进程问题
        pin_memory=False
    )
    
    return dataloader