# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 20:02:47 2020
https://blog.csdn.net/qq_36758914/java/article/details/105007428
@author: cofishe
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import random
import zipfile
import math


with zipfile.ZipFile('jaychou_lyrics.txt.zip') as zin:
    with zin.open('jaychou_lyrics.txt') as f:
        corpus_chars = f.read().decode('utf-8')
corpus_chars[:40]

corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')   #换行符替换成空格

corpus_chars = corpus_chars[0:10000]                                #取前1w字符

# test
#corpus_chars = [1,2,2,3,4,5,6,6,6,6,7,8,8,8,8,8,8,9]

#删掉重复的            获得列表
idx_to_char = list(set(corpus_chars))
len(idx_to_char)

#将字符映射到引索      对应关系
char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])

# 不同变量的个数  -->  字典大小
vocab_size = len(idx_to_char)

# 将文件里所有字符替换为索引
# 日后分析都是在所有字符替换为索引的基础上进行分析的
corpus_indices = [char_to_idx[char] for char in corpus_chars]


# 截取前 20 个
sample = corpus_indices[:20]
# 将索引 替换 为原本的 含义
sample_date = [idx_to_char[idx] for idx in sample]
print('chars:', ''.join(sample_date))
print('indices:', sample)


"""
从数据里随机采样一个小批量
其中批量大小 batch_size 指每个小批量的样本数
num_steps 为每个样本所包含的时间步数

在随机采样中 每个样本是原始序列上任意截取的一段序列
相邻的两个随机小批量在原始序列上的位置不一定相邻
因此 我们无法用一个小批量最终时间步的隐藏状态来初始化下一个小批量的隐藏状态

在训练模型时，每次随机采样前都需要重新初始化隐藏状态
"""
# 让我们输入一个从0到29的连续整数的人工序列
# 设批量大小和时间步数分别为2和6
# 打印随机采样每次读取的小批量样本的输入 X 和标签 Y
# 可见,相邻的两个随机小批量在原始序列上的位置不一定相毗邻
def data_iter_random(corpus_indices, batch_size, num_steps, ctx=None):
    # num_examples 是指有多少个样本，此时 num_examples=29 // 6=4
    # 减1是因为输出的索引（标签）是相应输入的索引（样本）加1
    # 也就是说，如果不减1的话，一旦 X 取到 [24 25 26 27 28 29]，Y 就只能取 [25 26 27 28 29]
    # 因为 corpus_indices （即 my_seq）就只到29
    num_examples = (len(corpus_indices) - 1) // num_steps
    
    # epoch_size 是指遍历一遍数据集需要训练的次数
    # 因为训练一次输入的样本数为 batch_size=2，那么遍历一次需要的次数为 epoch_size=2
    epoch_size = num_examples // batch_size
    
    # example_indices 是一个包含着四个样本各自索引的列表，即 example_indices=[0, 1, 2, 3]
    example_indices = list(range(num_examples))
    
    # 打乱 example_indices 列表的顺序。
    random.shuffle(example_indices)
    #print('序列' , example_indices)
    
    # 这个函数返回的是从 pos 开始的长为 num_steps 的序列
    def _data(pos):
        return corpus_indices[pos: pos + num_steps]
    
    # 每次输出一批样本和标签
    for i in range(epoch_size):
        # 每次读取batch_size个随机样本
        i = i * batch_size
        
        # 从 i 开始切 切 batch_size 份
        # 计算样本的索引号
        batch_indices = example_indices[i: i + batch_size]
        
        X = [_data(j * num_steps) for j in batch_indices]
        Y = [_data(j * num_steps + 1) for j in batch_indices]
        
        # yield 是指将 np.array(X), np.array(Y) 迭代输出。也就是说,每次只输出一个批次的样本和标签。
        yield np.array(X, ctx), np.array(Y, ctx)

# # EX: 随机采样 不重复 
# print('随机采样')
# my_seq = list(range(30))    # 人工序列 0 ~ 29
# for X, Y in data_iter_random(my_seq, batch_size=2, num_steps=6):
#     print('X: ', X, '\nY:', Y, '\n')



# 相邻采样
#                              数据        每批大小   时间序列长度
def data_iter_consecutive(corpus_indices, batch_size, num_steps, ctx=None):
    # 将列表转换为数组
    corpus_indices = np.array(corpus_indices)
    # 样本数量
    data_len = len(corpus_indices)
    
    # 一个批次的长度  表明一个批次中有多少个字符
    batch_len = data_len // batch_size
    
    # 从头取 batch_size * batch_len 个元素 变换为 [batch_size,batch_len]
    indices = corpus_indices[0: batch_size*batch_len].reshape((batch_size, batch_len))
    
    #epoch_size 是指遍历一遍数据集需要训练的次数，因为训练一次输入的样本数为 batch_size=2
    #那么遍历一次需要的次数为 epoch_size=2
    epoch_size = (batch_len - 1) // num_steps
    
    #进一步将每个数组的前6个字符放到位于同一批次的两个样本中,后6个字符放到位于另一批次的两个样本中
    for i in range(epoch_size):
        i = i * num_steps
        X = indices[:, i: i + num_steps]
        Y = indices[:, i + 1: i + num_steps + 1]
        yield X, Y

# EX:
print('相邻采样')
my_seq = list(range(30))
for X, Y in data_iter_random(my_seq, batch_size=2, num_steps=6):
    print('X: ', X, '\nY:', Y, '\n')





#构造一个含单隐藏层、隐藏单元个数为256的循环神经网络层 rnn_layer，并对权重做初始化
"""
其中，rnn_layer 的输入形状为 (时间步数, 批量大小, 词典大小)，在前向计算后会分别返回输出和隐藏状态
其中输出指的是隐藏层在各个时间步上计算并输出的隐藏状态，它们通常作为后续输出层的输入
需要强调的是，该“输出”本身并不涉及输出层计算，形状为 (时间步数, 批量大小, 隐藏单元个数)
返回的隐藏状态指的是隐藏层在最后时间步的隐藏状态：当隐藏层有多层时，每一层的隐藏状态都会记录在该变量中

"""

num_hiddens = 256   # 神经元个数
cell = keras.layers.SimpleRNNCell(num_hiddens, 
                                  kernel_initializer='glorot_uniform')
rnn_layer = keras.layers.RNN(cell,time_major=True,
                            return_sequences=True,return_state=True)


batch_size = 32      # 批量大小
state = rnn_layer.cell.get_initial_state(batch_size=batch_size,dtype=tf.float32)
print('State : ',state.shape)  # (32, 256)

num_steps = 35      # 时间步数

# 时间步数, 批量大小, 词典大小
X = tf.random.uniform(shape=(num_steps, batch_size, vocab_size))

print(X.shape)  #    (35, 32, 1027)

# 输出  输出状态
Y, state_new = rnn_layer(X, state)
print(Y.shape)  #    (35, 32, 256)

#print('state_new : ', state_new)
print(len(state_new))      #   32
print(state_new[0].shape)  # (256,)


"""
输入形状为 (批量大小, 时间步数)
将输入转置成 (时间步数, 批量大小)
利用独热编码得到 (时间步数, 批量大小, 词典大小)
输入到 rnn 层，得到 (时间步数, 批量大小, 隐藏单元个数)
reshape 成 (时间步数x批量大小, 隐藏单元个数)
经过 Dense 层，得到 (时间步数x批量大小, 词典大小)

"""
class RNNModel(tf.keras.Model):
    def __init__(self, rnn_layer, vocab_size):
        super().__init__()
        self.rnn = rnn_layer
        self.vocab_size = vocab_size    # 词典大小
        self.dense = keras.layers.Dense(vocab_size)
        
    def call(self, inputs, state):
        # 将输入转置成(num_steps, batch_size)后获取one-hot向量表示
        X = tf.one_hot(tf.transpose(inputs), self.vocab_size)
        Y, state = self.rnn(X, state)
        # 全连接层会首先将Y的形状变成(num_steps * batch_size, num_hiddens)
        # 它的输出形状为(num_steps * batch_size, vocab_size) 时间步长,批尺寸,词典大小
        output = self.dense(tf.reshape(Y, (-1, Y.shape[-1])))
        return output, state
    
    # 获取初始状态
    def get_initial_state(self, *args, **kwargs):
        return self.rnn.cell.get_initial_state(*args, **kwargs)

# 
model = RNNModel(rnn_layer, vocab_size)

# 预测函数
def predict_rnn_keras(prefix, num_chars):
    # 使用model的成员函数来初始化隐藏状态
    state = model.get_initial_state(batch_size=1,dtype=tf.float32)
    output = [char_to_idx[prefix[0]]]
    
    for t in range(num_chars + len(prefix) - 1):
        X = np.array([output[-1]]).reshape((1, 1))
        Y, state = model(X, state)  # 前向计算不需要传入模型参数
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            output.append(int(np.array(tf.argmax(Y,axis=-1))))

    return ''.join([idx_to_char[i] for i in output])


# 裁剪梯度 
# 类似给梯度乘了个台子分布
#   min(theta/ads(grads), 1) * grads
def grad_clipping(grads,theta):
    # 计算 grads 的 均值
    norm = np.array([0])
    for i in range(len(grads)):
        norm+=tf.math.reduce_sum(grads[i] ** 2)
    norm = np.sqrt(norm).item()
    
    new_gradient=[]
    if norm > theta:
        # theta/ads(grads) * grads
        for grad in grads:
            new_gradient.append(grad * theta / norm)
    else:
        # grads
        for grad in grads:
            new_gradient.append(grad)  
    return new_gradient

lr = 1e2
optimizer=tf.keras.optimizers.SGD(learning_rate=lr)

# 定义梯度下降函数
#  默认裁剪阈值0.01
def train_step(X, state, Y, clipping_theta=1e-2):
    with tf.GradientTape(persistent=True) as tape:
        (outputs, state) = model(X, state)
        
        # 将输出化为列向量
        y = Y.T.reshape((-1,))
        
        # 计算损失  l = loss_object(y,outputs)
        l = tf.reduce_mean(tf.losses.sparse_categorical_crossentropy(y, outputs))

    # 计算梯度
    grads = tape.gradient(l, model.variables)
    #print(grads)  (1027,)
    
    # 梯度裁剪
    grads = grad_clipping(grads, clipping_theta)
    optimizer.apply_gradients(zip(grads, model.variables))  # 因为已经误差取过均值，梯度不用再做平均
    return l, y

# 训练                            训练次数     批大小      显示间隔    预测长度    输入
def train_and_predict_rnn_keras(num_epochs, batch_size, pred_period, pred_len, prefixes):
    for epoch in range(num_epochs):
        l_sum, n = 0.0, 0
        # 获得批处理数据
        data_iter = data_iter_consecutive(
            corpus_indices, batch_size, num_steps)
        
        # 获得上一次的状态  (batch_size, num_hiddens)
        state = model.get_initial_state(batch_size=batch_size,dtype=tf.float32)
        
        # 获得损失和 l_sum
        # 获取计算的损失总样本数 n = num_hiddens * num_steps = batch_size * 迭代一次循环次数 * num_steps
        # n = 8960  --> 1120 * 8 --> 32 * 35 * 8
        # 为什么是循环 8 次 : 256 / 32 = 8 --> 256个神经元没有一次求出 --> 正确！！
        for X, Y in data_iter:
            # X.shape (32, 35)
            # Y.shape (32, 35)
            # 根据输入 上一次状态 label  计算输出与损失
            
            l, y = train_step(X, state, Y)
            l_sum += np.array(l).item() * len(y)
            
            n += len(y)
        
        # 应该显示的地方
        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f' % (
                epoch + 1, math.exp(l_sum / n)))
            for prefix in prefixes:
                # 输出预测 
                print(' -', predict_rnn_keras(prefix, pred_len))




num_epochs, batch_size = 3000, 256
pred_period, pred_len, prefixes = 50, 15, ['分开', '朝海']
train_and_predict_rnn_keras(num_epochs, batch_size, pred_period,
                            pred_len, prefixes)



