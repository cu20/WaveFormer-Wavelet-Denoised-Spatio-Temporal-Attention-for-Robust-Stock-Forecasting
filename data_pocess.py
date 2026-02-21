import pandas as pd
import tensorflow as tf 
import numpy as np 
import pandas as pd 
import pickle 
import argparse
from tqdm import tqdm
import os 
import psutil
path = './datasets/CSI300_dataset_2020-01-01_2020-09-25.csv'

attributes = ['open', 'high', 'low', 'close', 'volume', 'vwap', 'return_1d', 
            'return_5d', 'return_20d', 'high_low_ratio', 'price_range', 'ma5', 
            'ma10', 'ma20', 'ma30', 'ma60', 'price_strength', 'price_position', 'volume_ma5', 
            'volume_ma20', 'volume_ratio', 'price_volume_corr', 'volatility_5d', 'volatility_20d',
             'momentum_5d', 'momentum_20d', 'relative_strength', 'market_return', 'market_return_std', 
             'market_avg_close', 'market_close_std', 'market_total_volume', 'market_avg_volume', 'market_breadth', 
             'market_volatility', 'label_1d', 'label_5d', 'label_20d']
# 短特征列表：用于模型输入的特征列（不应包含标签列）
short_att = ['open', 'high', 'low', 'close', 'volume', 'vwap', 'return_1d', 'return_5d', 'return_20d', 'high_low_ratio', 'price_range', 'ma5', 'ma10', 'ma20', 'ma30', 'ma60', 'price_strength', 'price_position', 'volume_ma5', 'volume_ma20', 'volume_ratio', 'price_volume_corr', 'volatility_5d', 'volatility_20d', 'momentum_5d', 'momentum_20d', 'relative_strength', 'market_return', 'market_return_std', 'market_avg_close', 'market_close_std', 'market_total_volume', 'market_avg_volume', 'market_breadth', 'market_volatility']

# 标签列：这里选择 5 日收益率标签
return_column = 'label_5d'

def process_data(file_path:str, list_attr:list,train_perc:float, thresh=10, time_periods=5):
    num_stocks = 0
    train = []
    valid = []
    test = []
    a = 1
    for total_stock in pd.read_csv(file_path, chunksize=400000, memory_map=True):
        print(f'开始处理第 {a} 个区块')
        # 确保 instrument / datetime / 标签列 存在，即使不在特征列表中也要保留
        required_cols = ['instrument']
        if 'datetime' in total_stock.columns:
            required_cols.append('datetime')
        # 标签列（如 label_5d）也必须加载进来
        if return_column in total_stock.columns:
            required_cols.append(return_column)
        # 合并必需列和特征列（去重）
        cols_to_keep = list(set(list_attr + required_cols))
        shortended_df = total_stock[cols_to_keep]
        # 填充任何NaN值，确保每个观察都有所有必要的数据字段
        shortended_df = shortended_df.fillna(0)
        # 按股票ID分组
        groups = shortended_df.groupby(['instrument'])
        unique_ids = shortended_df['instrument'].unique().tolist()
        # 创建一个列表来存储每个股票的观察数量
        cutoff = []
        # 遍历每个股票ID
        for i in tqdm(range(len(unique_ids))):
            num = unique_ids[i]
            # 获取每个股票ID的观察数量
            count = groups.get_group((num,))['return_1d'].count()
            # 如果观察数量大于等于阈值，则将股票ID添加到cutoff列表中
            if count >= thresh:
                cutoff.append(num)
        print(f'股票截断完成并选择了 {len(cutoff)} 只股票 ')

        
        print("进入股票切片部分")
        for i in tqdm(range(len(cutoff))):
          # 获取当前股票ID
            num = cutoff[i]
            # curr_df 是一个 (时间周期, 特征长度) 矩阵，其中时间周期 >= 阈值
            curr_df = groups.get_group((num,))
            # 增加股票数量计数
            num_stocks+=1
            # 使用完整数据，按 train_perc 比例划分训练/验证/测试集
            curr_df_full = curr_df.copy()
            total_steps = len(curr_df_full)
            
            # 确保至少有 thresh 个时间步用于训练/验证
            if total_steps < thresh:
                # 如果数据太少，跳过这只股票
                continue
            
            # 按 train_perc 计算训练集结束位置（基于完整数据）
            index_train = int(np.floor(total_steps * train_perc))
            # 确保训练+验证区间长度是 time_periods 的整数倍
            index_train = index_train - (index_train % time_periods)
            
            # 验证集：使用训练集之后的一小部分（例如10%或至少 time_periods 个时间步）
            # 计算验证集大小：取剩余数据的20%或至少 time_periods 个时间步
            remaining_after_train = total_steps - index_train
            valid_size = max(time_periods, int(np.floor(remaining_after_train * 0.2)))
            valid_size = valid_size - (valid_size % time_periods)  # 确保是 time_periods 的倍数
            index_valid = index_train + valid_size
            
            # 测试集：剩余的所有数据
            # 但为了确保训练集有足够的数据，如果剩余数据太少，调整验证集大小
            if remaining_after_train < time_periods * 2:
                # 如果剩余数据太少，不单独划分验证集，全部作为测试集
                index_valid = index_train
                valid_size = 0
            
            # 获取收益率（标签列）
            returns_full = curr_df_full[return_column].to_numpy()
            
            # 删除日期时间列、instrument列以及标签列，保留纯特征
            cols_to_drop = []
            if 'datetime' in curr_df_full.columns:
                cols_to_drop.append('datetime')
            if 'instrument' in curr_df_full.columns:
                cols_to_drop.append('instrument')
            # 删除所有标签列，避免被当作特征
            for lbl in ['label_1d', 'label_5d', 'label_20d']:
                if lbl in curr_df_full.columns:
                    cols_to_drop.append(lbl)
            obs_full = curr_df_full.drop(columns=cols_to_drop).to_numpy() if cols_to_drop else curr_df_full.to_numpy()

            # TODO:标准化特征和收益率
            returns_full:tf.Tensor = tf.math.divide(returns_full, 1)
            # TODO:标准化特征
            feature_full:tf.Tensor = tf.math.divide(obs_full, 1)

            # 训练集：前 index_train 个时间步
            train_feature = feature_full[:index_train, :]
            train_returns = returns_full[:index_train]
            train_num = train_returns.shape[0]
            train_iters = int(np.floor(train_num / time_periods))
            
            # 添加训练数据
            for i in range(0, train_iters):
                start = i * time_periods
                end = start + time_periods
                curr_feat = train_feature[start:end, :]
                curr_ret = train_returns[start:end]
                train.append((curr_feat, curr_ret))
            
            # 验证集：从 index_train 到 index_valid，按 time_periods 分块（与训练/测试集一致）
            if valid_size > 0:
                validate_feature = feature_full[index_train:index_valid, :]
                validate_returns = returns_full[index_train:index_valid]
                valid_num = validate_returns.shape[0]
                valid_iters = int(np.floor(valid_num / time_periods))
                for k in range(0, valid_iters):
                    start = k * time_periods
                    end = start + time_periods
                    curr_feat = validate_feature[start:end, :]
                    curr_ret = validate_returns[start:end]
                    valid.append((curr_feat, curr_ret))

            # 测试集：从 index_valid 开始的所有剩余数据
            test_feature = feature_full[index_valid:, :]
            test_rets = returns_full[index_valid:]
            test_num = test_rets.shape[0]
            test_iters = int(np.floor(test_num/time_periods))
            
            
            # 测试集：将测试集分为多个样本，每个样本包含 time_periods 个时间步
            # 使用 j 作为当前测试样本的索引，避免与训练循环中的 i 混淆
            for j in range(0, test_iters):
                start = j * time_periods
                end = start + time_periods
                feat_test = test_feature[start:end, :]
                ret_test = test_rets[start:end]
                test.append((feat_test, ret_test))
        print(f'第 {a} 个区块处理完成，股票数量 {num_stocks}')
        a+=1
       
    
    return train, valid, test

def main():
    #解析命令行参数
    parser = argparse.ArgumentParser()
    # 添加文件名参数
    parser.add_argument('-f', '--file_name', required=True)
    # 解析参数
    arg = vars(parser.parse_args())
    file_name = arg['file_name']

    # 调用 clean function 
    train_data, valid_data, test_data = process_data(file_name, short_att, 0.8)
    
    # 创建 data 目录（如果不存在）
    os.makedirs('data', exist_ok=True)
    
    # 保存训练集
    with open('data/pickle_train', 'wb') as file:
        pickle.dump(train_data, file)
    # 保存验证集
    with open('data/pickle_validate', 'wb') as file:
        pickle.dump(valid_data, file)
    # 保存测试集
    with open('data/pickle_test', 'wb') as file:
        pickle.dump(test_data, file)
    
    print(f'\n数据保存完成！')
    print(f'  - 训练集样本数: {len(train_data)}')
    print(f'  - 验证集样本数: {len(valid_data)}')
    print(f'  - 测试集样本数: {len(test_data)}')
        
    
if __name__ == "__main__":
    main()



