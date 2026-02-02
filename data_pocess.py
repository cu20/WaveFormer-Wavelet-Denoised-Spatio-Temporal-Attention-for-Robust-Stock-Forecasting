import pandas as pd
import tensorflow as tf 
import numpy as np 
import pandas as pd 
import pickle 
import argparse
from tqdm import tqdm
import os 
import psutil
path = '../Downloads/total_data.csv'

attributes = []
short_att = []

def process_data(file_path:str, list_attr:list,train_perc:float, thresh=10, time_periods=4):
    num_stocks = 0
    train = []
    valid = []
    test = []
    a = 1
    for total_stock in pd.read_csv(file_path, chunksize=400000, memory_map=True):
        print(f'开始处理第 {a} 个区块')
        #只保留普通股和主交易所的股票
        total_stock = total_stock.loc[(total_stock['common']==1) & total_stock['exch_main']==1]
        #只保留指定的特征
        shortended_df = total_stock[list_attr]
        # 填充任何NaN值，确保每个观察都有所有必要的数据字段
        shortended_df = shortended_df.fillna(0)
        # 按股票ID分组
        groups = shortended_df.groupby(['id'])
        unique_ids = shortended_df['id'].unique().tolist()
        # 创建一个列表来存储每个股票的观察数量
        cutoff = []
        # 遍历每个股票ID
        for i in tqdm(range(len(unique_ids))):
            num = unique_ids[i]
            # 获取每个股票ID的观察数量
            count = groups.get_group(num)['ret'].count()
            # 如果观察数量大于等于阈值，则将股票ID添加到cutoff列表中
            if count >= thresh:
                cutoff.append(num)
        print(f'股票截断完成并选择了 {len(cutoff)} 只股票 ')

        
        print("进入股票切片部分")
        for i in tqdm(range(len(cutoff))):
          # 获取当前股票ID
            num = cutoff[i]
            # curr_df 是一个 (时间周期, 特征长度) 矩阵，其中时间周期 >= 阈值
            curr_df = groups.get_group(num)
            # 获取当前股票的收益率
            returns  = curr_df['ret_exc_lead1m'].to_numpy()
            # 增加股票数量计数
            num_stocks+=1
            # 修剪 curr_df 使得所有股票都有相同的回看周期
            curr_df = curr_df.iloc[0:thresh, :]
            # 删除日期和收益率列，并转换为numpy数组
            obs = curr_df.drop(columns=['ret_exc_lead1m','date']).to_numpy()

            # TODO:标准化特征和收益率
            feature_max = tf.math.reduce_max(obs, axis=0)
            return_max  = tf.math.reduce_max(returns)
            returns:tf.Tensor = tf.math.divide(returns, 1)
            # TODO:标准化特征
            feature:tf.Tensor= tf.math.divide(obs, 1)

            # 将数据分为训练集、验证集和测试集，train_perc为训练集比例，np.floor为向下取整
            # feature.shape[0]为总的时间步数，time_periods为一个样本的时间步数
            index = int(np.floor(feature.shape[0]*train_perc))
            # 确保训练+验证区间长度是 time_periods 的整数倍
            index = index - (index % time_periods)

            # 预留最后一个 time_periods 作为验证集，其余作为训练集
            if index <= time_periods:
                # 序列太短就不单独划验证集了，全部作为训练
                train_end = index
                valid_start = 0
            else:
                # 训练集结束位置为总时间步数减去时间步数，验证集开始位置为训练集结束位置
                train_end = index - time_periods
                valid_start = train_end

            # 计算训练集的迭代次数
            iters = int(train_end / time_periods)
            # 训练集特征和收益率
            train_feature = feature[:train_end, :]
            train_returns = returns[:train_end]
            
            # 添加训练数据
            for i in range(0, iters):
                # 获取当前样本的特征和收益率
                curr_feat = feature[i*time_periods:time_periods + i*time_periods,:]
                curr_ret = returns[i*time_periods:time_periods + i*time_periods]
                # 添加当前样本到训练集，格式为(特征[time_periods,特征数量],收益率[time_periods])
                train.append((curr_feat,curr_ret))
            
            # 验证集：使用紧挨着训练集末尾的一个窗口，避免与训练样本重叠
            if index >= time_periods:
                validate_feature = feature[valid_start:index, :]
                validate_returns = returns[valid_start:index]
                valid.append((validate_feature, validate_returns))

            # 测试集：使用剩余的所有时间步
            test_feature  =feature[index:,:]
            test_rets = returns[index:]
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
        print(f'{chunk_size*a} 条数据处理完成，股票数量 {num_stocks}')
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
    # 保存训练集
    with open('data/pickle_train', 'wb') as file:
        pickle.dump(train_data,file )
        file.close()
    # 保存验证集
    with open('data/pickle_validate', 'wb') as file:
        pickle.dump(valid_data, file); 
        file.close()
    # 保存测试集
    with open('data/pickle_test', 'wb') as file:
        pickle.dump(test_data, file)
        file.close()
        
    
if __name__ == "__main__":
    main()



