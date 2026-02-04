import numpy as np
import pandas as pd
import tensorflow as tf
import random

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

# 创建用于训练的数据库(TensorFlow数据集)
# data是包含特征和收益率的元组列表，shuffle为是否打乱数据，batch_size为批量大小
def create_tf_dataset(data, shuffle=False, batch_size= 400):

    # 创建一个形状为(数据长度, 时间步长,特征数量+1)的数组，其中最后一个维度是收益率
    data_arr = np.zeros(shape=(len(data), data[0][0].shape[0], features_num+1))
    # 创建一个空列表来存储最终的批次
    final_batches = []
    
    # 遍历每个数据样本
    for i in range(0, len(data)):
        # 将特征转换为numpy数组
        nparr = data[i][0].numpy()
        # 将特征复制到data_arr中
        data_arr[i,:, 0:features_num] = nparr
        # 将收益率复制到data_arr中
        data_arr[i,:,features_num] = data[i][1].numpy()

    if shuffle == True:
        # 如果需要打乱数据，则打乱data_arr
        np.random.shuffle(data_arr)
        # 创建一个形状为(数据长度)的数组来存储收益率
        returns_arr  = np.zeros(shape=(len(data)))
        # 遍历每个数据样本
        for i in range(0, len(data)):
            # 将收益率复制到returns_arr中
            returns_arr[i] = data_arr[i,data[0][0].shape[0]-1,features_num]
        # 删除收益率列
        data_arr = data_arr[:,:,0:features_num]

        # 准备数据切片
        stocks_left = data_arr.shape[0] - batch_size
        i =0 
        while (stocks_left >= batch_size):
            # 获取当前批次的特征
            curr_feature = data_arr[i*batch_size: batch_size + i*batch_size,:,0:features_num]
            # 获取当前批次的收益率
            curr_returns = returns_arr[i*batch_size: batch_size + i*batch_size]
            # 将当前批次添加到最终批次列表中
            final_batches.append((tf.convert_to_tensor(curr_feature), tf.convert_to_tensor(curr_returns)))
            # 增加索引
            i+=1 
            # 减少剩余股票数量
            stocks_left = stocks_left - batch_size
        
        if stocks_left < batch_size and stocks_left > 5:
            # 获取当前批次的长度
            curr_len = stocks_left
            # 获取最后一个条目的索引，最后一个条目是当前批次的长度减去剩余股票数量
            last_entry =i*batch_size +batch_size
            # 获取当前批次的长度
            curr_feature = data_arr[last_entry :last_entry+curr_len,:,0:features_num]
            # 获取当前批次的长度
            curr_returns = returns_arr[last_entry :last_entry+curr_len]
            # 将当前批次添加到最终批次列表中
            curr_returns = returns_arr[last_entry :last_entry+curr_len]
            final_batches.append((tf.convert_to_tensor(curr_feature), tf.convert_to_tensor(curr_returns)))
        # 返回最终批次列表
        return final_batches
    # 如果不需要打乱数据，则直接返回数据
    else:
        # 创建一个形状为(数据长度)的数组来存储收益率
        returns_arr  = np.zeros(shape=(len(data)))
        # 遍历每个数据样本
        for i in range(0, len(data)):
            # 将收益率复制到returns_arr中
            returns_arr[i] = data_arr[i,4,features_num]
        data_arr = data_arr[:,:,0:features_num]
        # 准备数据切片
        while (stocks_left >= batch_size):
            # 获取当前批次的长度
            curr_len = batch_size
            # 获取最后一个条目的索引，最后一个条目是当前批次的长度减去剩余股票数量
            last_entry =i*batch_size +batch_size
            # 获取当前批次的长度
            curr_feature = data_arr[i*batch_size: batch_size + i*batch_size,:,0:features_num]
            curr_returns = returns_arr[i*batch_size: batch_size + i*batch_size]
            final_batches.append((tf.convert_to_tensor(curr_feature), tf.convert_to_tensor(curr_returns)))
            i+=1 
            stocks_left = stocks_left - batch_size
        
        if stocks_left < batch_size and stocks_left > 5:
            curr_len = stocks_left
            last_entry =i*batch_size +batch_size
            curr_feature = data_arr[last_entry :last_entry+curr_len,:,0:features_num]
            curr_returns = returns_arr[last_entry :last_entry+curr_len]
            final_batches.append((tf.convert_to_tensor(curr_feature), tf.convert_to_tensor(curr_returns)))
        
        return final_batches


class DailyBatchSamplerRandom:
    def __init__(self):
        pass