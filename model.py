import numpy as np
import tensorflow as tf
from tensorflow import keras
import math

from base_model import SequenceModel

class PositionalEncoding(keras.layers.Layer):
    def __init__(self, d_model, max_len=100, **kwargs):
        super().__init__(**kwargs)
        pos_encoding = np.zeros((max_len, d_model))
        position = np.arange(0, max_len, dtype=np.float32)[:, np.newaxis]
        #位置编码公式
        div_term = np.exp(np.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pos_encoding[:, 0::2] = np.sin(position * div_term)
        pos_encoding[:, 1::2] = np.cos(position * div_term)
        #将位置编码矩阵转为常量张量
        self.pos_encoding = tf.constant(pos_encoding, dtype=tf.float32)
    
    def call(self, inputs):
        #直接返回输入张量和位置编码的和
        return inputs + self.pos_encoding[:tf.shape(inputs)[1], :]

# 股票间聚合（Spatial-Attention）
class SAttention(keras.layers.Layer):
    def __init__(self, d_model, num_heads, dropout_rate, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model
        # 计算每个头的维度大小,每个Head负责d_model的一部分
        self.dim = d_model // num_heads
        # 处理无法被整除的情况
        self.remainder = d_model % num_heads

        self.dropout_rate = dropout_rate
        #将输入映射到512*512的空间中的全连接层，生成的矩阵是可学习矩阵
        self.q_dense = keras.layers.Dense(d_model, use_bias=False)
        self.k_dense = keras.layers.Dense(d_model, use_bias=False)
        self.v_dense = keras.layers.Dense(d_model, use_bias=False)
        #生成一个列表，为每个头分配dropout层
        self.attn_dropout_layers = [keras.layers.Dropout(dropout_rate) for _ in range(num_heads)]
        #epsilon是防止除零误差的偏置
        self.norm1 = keras.layers.LayerNormalization(epsilon=1e-5)
        self.norm2 = keras.layers.LayerNormalization(epsilon=1e-5)
        
        self.ffn = keras.Sequential([
            keras.layers.Dense(d_model, activation='relu'),
            keras.layers.Dropout(dropout_rate),
            keras.layers.Dense(d_model),
            keras.layers.Dropout(dropout_rate)
        ])
    
    def call(self, inputs, training=None):
        # inputs shape: (N, T, D)

        normed_inputs = self.norm1(inputs)

        q = tf.transpose(self.q_dense(normed_inputs), perm=[1, 0, 2])
        k = tf.transpose(self.k_dense(normed_inputs), perm=[1, 0, 2])
        v = tf.transpose(self.v_dense(normed_inputs), perm=[1, 0, 2])
        # q, k, v shape: (T, N, D)
        #T:seq_len, N:batch_size, D:d_model 

        attention_output = []
        cur_idx = 0
        for i in range(self.num_heads):
            #用前面的头进行补齐维度
            if i < self.remainder:
                cur_dim = self.dim + 1
            else:
                cur_dim = self.dim
            #根号d_k
            temperature = math.sqrt(cur_dim)
            end_idx = cur_idx + cur_dim
            q_i = q[:, :, cur_idx:end_idx]
            k_i = k[:, :, cur_idx:end_idx]
            v_i = v[:, :, cur_idx:end_idx]
            # shape: (T, N, D/N) or (T, N, D/N + 1)
            #Q*K^T,(T,N,D)*(T,D,N) = (T, N, N)
            attention_weights = tf.matmul(q_i, k_i, transpose_b=True) / temperature  # (T, N, N)
            #在最后一个维度进行softmax归一化（当前股票对所有股票的注意力权重）
            attention_weights = tf.nn.softmax(attention_weights, axis=-1)
            #在训练时进行dropout
            if training:
                attention_weights = self.attn_dropout_layers[i](attention_weights, training=training)
            #将注意力权重与v相乘，得到当前股票的注意力输出
            attention_output.append(tf.transpose(tf.matmul(attention_weights, v_i), perm=[1, 0, 2])) 

            cur_idx = end_idx
        #将各个Head在d_model维度上所有股票的注意力输出拼接起来，得到最终的注意力输出
        concat_output = tf.concat(attention_output, axis=-1)  # (N, T, D)
        #将注意力输出与这一层输入进行残差连接，并进行LayerNormalization
        attention_output = self.norm2(concat_output + inputs)
        #将注意力输出通过FFN层，得到最终的注意力输出
        ffn_output = self.ffn(attention_output, training=training)

        return attention_output + ffn_output

#股票内聚合(Temporal Attention)
class TAttention:
    def __init__(self, d_model, num_heads, dropout_rate, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model
        #处理几乎同上，只是不用transpose交换维度了
        self.dim = d_model // num_heads

        self.remainder = d_model % num_heads
        self.dropout_rate = dropout_rate

        self.q_dense = keras.layers.Dense(d_model, use_bias=False)
        self.k_dense = keras.layers.Dense(d_model, use_bias=False)
        self.v_dense = keras.layers.Dense(d_model, use_bias=False)

        self.attn_dropout_layers = [keras.layers.Dropout(dropout_rate) for _ in range(num_heads)]

        self.norm1 = keras.layers.LayerNormalization(epsilon=1e-5)
        self.norm2 = keras.layers.LayerNormalization(epsilon=1e-5)

        self.ffn = keras.Sequential([
            keras.layers.Dense(d_model, activation='relu'),
            keras.layers.Dropout(dropout_rate),
            keras.layers.Dense(d_model),
            keras.layers.Dropout(dropout_rate)
        ])

    def call(self, inputs, training=None):
        # inputs shape: (N, T, D)

        normed_inputs = self.norm1(inputs)

        q = self.q_dense(normed_inputs)
        k = self.k_dense(normed_inputs)
        v = self.v_dense(normed_inputs)
        # q, k, v shape: (N, T, D)

        attention_output = []
        cur_idx = 0
        for i in range(self.num_heads):
            # different implementation of head dimension than the original paper
            if i < self.remainder:
                cur_dim = self.dim + 1
            else:
                cur_dim = self.dim

            end_idx = cur_idx + cur_dim
            q_i = q[:, :, cur_idx:end_idx]
            k_i = k[:, :, cur_idx:end_idx]
            v_i = v[:, :, cur_idx:end_idx]
            # shape: (N, T, D/N) or (N, T, D/N + 1)

            attention_weights = tf.matmul(q_i, k_i, transpose_b=True) # No temperature scaling
            attention_weights = tf.nn.softmax(attention_weights, axis=-1)

            if training:
                attention_weights = self.attn_dropout_layers[i](attention_weights, training=training)

            attention_output.append(tf.matmul(attention_weights, v_i))

            cur_idx = end_idx
        
        concat_output = tf.concat(attention_output, axis=-1)  # (N, T, D)
        attention_output = self.norm2(concat_output + inputs)
        ffn_output = self.ffn(attention_output, training=training)

        return attention_output + ffn_output