# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 09:33:07 2020

@author: 爬上阁楼的鱼
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

#print(corpus_chars)

#删掉重复的   获得列表
idx_to_char = list(set(corpus_chars))
len(idx_to_char)

#将字符映射到引索
char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])

#打印印索观察
#print(char_to_idx)

#得到字典
vocab_size = len(char_to_idx)
#print(vocab_size)

#将歌词文件里所有字符转化为印索
corpus_indices = [char_to_idx[char] for char in corpus_chars]
#print(corpus_indices)
#print(len(corpus_indices))

#
sample = corpus_indices[:20]
print('chars:', ''.join([idx_to_char[idx] for idx in sample]))
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
# 让我们输入一个从0到29的连续整数的人工序列。设批量大小和时间步数分别为2和6
#打印随机采样每次读取的小批量样本的输入 X 和标签 Y
#可见，相邻的两个随机小批量在原始序列上的位置不一定相毗邻
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
    
    # 这个函数返回的是从 pos 开始的长为 num_steps 的序列
    def _data(pos):
        return corpus_indices[pos: pos + num_steps]

    # 
    for i in range(epoch_size):
        # 每次读取batch_size个随机样本
        i = i * batch_size
        batch_indices = example_indices[i: i + batch_size]
        X = [_data(j * num_steps) for j in batch_indices]
        Y = [_data(j * num_steps + 1) for j in batch_indices]
        
        # yield 是指将 np.array(X), np.array(Y) 迭代输出。也就是说，每次只输出一个批次的样本和标签。
        yield np.array(X, ctx), np.array(Y, ctx)
        

my_seq = list(range(30))
for X, Y in data_iter_random(my_seq, batch_size=2, num_steps=6):
    print('X: ', X, '\nY:', Y, '\n')


# 相邻采样
def data_iter_consecutive(corpus_indices, batch_size, num_steps, ctx=None):
    #将列表转换为数组
    corpus_indices = np.array(corpus_indices)
    #样本数量
    data_len = len(corpus_indices)
    
    #表明一个批次中有多少个字符。
    batch_len = data_len // batch_size
    
    indices = corpus_indices[0: batch_size*batch_len].reshape((
        batch_size, batch_len))
    
    #epoch_size 是指遍历一遍数据集需要训练的次数，因为训练一次输入的样本数为 batch_size=2
    #那么遍历一次需要的次数为 epoch_size=2
    epoch_size = (batch_len - 1) // num_steps
    
    #进一步将每个数组的前6个字符放到位于同一批次的两个样本中；后6个字符放到位于另一批次的两个样本中
    for i in range(epoch_size):
        i = i * num_steps
        X = indices[:, i: i + num_steps]
        Y = indices[:, i + 1: i + num_steps + 1]
        yield X, Y

# test
my_seq = list(range(30))
for X, Y in data_iter_random(my_seq, batch_size=2, num_steps=6):
    print('X: ', X, '\nY:', Y, '\n')


def to_onehot(X, size):
    # X shape: (batch, steps), output shape: (batch, vocab_size)
    return [tf.one_hot(x, size,dtype=tf.float32) for x in X.T]

#ex:
#X = np.arange(10).reshape((2, 5))
#inputs = to_onehot(X, vocab_size)



num_epochs = 2500  # 训练2500次
num_steps = 35  # 时间步长为35

# 输入尺寸    神经元数量    输出尺寸   =  词典大小    
num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size

#    获取参数
def get_params():
    def _one(shape):
        return tf.Variable(tf.random.normal(shape=shape,
                                             stddev=0.01,
                                             mean=0,
                                             dtype=tf.float32))

    # 隐藏层参数
    W_xh = _one((num_inputs, num_hiddens))
    W_hh = _one((num_hiddens, num_hiddens))
    b_h = tf.Variable(tf.zeros(num_hiddens), dtype=tf.float32)
    # 输出层参数
    W_hq = _one((num_hiddens, num_outputs))
    b_q = tf.Variable(tf.zeros(num_outputs), dtype=tf.float32)
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    return params

#初始化为隐藏状态
def init_rnn_state(batch_size):
    return tf.zeros(shape=(batch_size, num_hiddens))


def rnn(inputs, state, params):
    W_xh, W_hh, b_h, W_hq, b_q = params
    H = state
    outputs = []
    for X in inputs:
        X = tf.reshape(X,[-1,W_xh.shape[0]])
        H = tf.tanh(tf.matmul(X, W_xh) + tf.matmul(H, W_hh) + b_h)
        Y = tf.matmul(H, W_hq) + b_q
        outputs.append(Y)
    return outputs, H

#ex:
# A = np.array([[1, 2, 3, 4, 5, 6]])
# print(A.shape)
# for a in A:
#     print(a.shape)



#预训练函数
def predict_rnn(prefix, num_chars, params):
    state = init_rnn_state(batch_size=1)
    output = [char_to_idx[prefix[0]]]
    for t in range(num_chars + len(prefix) - 1):
        # 将上一时间步的输出作为当前时间步的输入
        X = tf.convert_to_tensor(to_onehot(np.array([output[-1]]), vocab_size),dtype=tf.float32)
        # 计算输出和更新隐藏状态
        (Y, state) = rnn(X, state, params)
        # 下一个时间步的输入是prefix里的字符或者当前的最佳预测字符
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            output.append(int(np.array(tf.argmax(Y[0],axis=1))))
    return ''.join([idx_to_char[i] for i in output])


#ex:
#params = get_params()
#print(predict_rnn('分开', 10, params))


# 计算裁剪后的梯度
def grad_clipping(grads,theta):
    norm = np.array([0])
    for i in range(len(grads)):
        norm+=tf.math.reduce_sum(grads[i] ** 2)
    norm = np.sqrt(norm).item()
    new_gradient=[]
    if norm > theta:
        for grad in grads:
            new_gradient.append(grad * theta / norm)
    else:
        for grad in grads:
            new_gradient.append(grad)  
    return new_gradient


#初始化优化器
optimizer = tf.keras.optimizers.SGD(learning_rate=1e2)


#定义梯度下降函数
def train_step(params, X, Y, state, clipping_theta):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(params)
        inputs = to_onehot(X, vocab_size)
        # outputs有num_steps个形状为(batch_size, vocab_size)的矩阵
        (outputs, state) = rnn(inputs, state, params)
        # 拼接之后形状为(num_steps * batch_size, vocab_size)
        outputs = tf.concat(outputs, 0)
        # Y的形状是(batch_size, num_steps)，转置后再变成长度为
        # batch * num_steps 的向量，这样跟输出的行一一对应
        y = Y.T.reshape((-1,))
        y = tf.convert_to_tensor(y, dtype=tf.float32)
        # 使用交叉熵损失计算平均分类误差
        l = tf.reduce_mean(tf.losses.sparse_categorical_crossentropy(y, outputs))

    # 根据 损失 与 参数 获取 梯度
    grads = tape.gradient(l, params)
    grads = grad_clipping(grads, clipping_theta)  # 裁剪梯度
    optimizer.apply_gradients(zip(grads, params))
    return l, y


# 定义训练函数
def train_and_predict_rnn(is_random_iter, batch_size, clipping_theta, pred_period, pred_len, prefixes):
    if is_random_iter:
        data_iter_fn = data_iter_random
    else:
        data_iter_fn = data_iter_consecutive
    params = get_params()
    
    for epoch in range(num_epochs):
        if not is_random_iter:  # 如使用相邻采样，在epoch开始时初始化隐藏状态
            state = init_rnn_state(batch_size)
        l_sum, n = 0.0, 0
        
        # [X,Y]                       印索          批大小     时间步长
        data_iter = data_iter_fn(corpus_indices, batch_size, num_steps)
        for X, Y in data_iter:
            if is_random_iter:  # 如使用随机采样，在每个小批量更新前初始化隐藏状态
                state = init_rnn_state(batch_size, num_hiddens) # state : batch_size x num_hiddens
            l, y = train_step(params, X, Y, state, clipping_theta)
            l_sum += np.array(l).item() * len(y)
            n += len(y)

        # 打印预测
        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f' % (epoch + 1, math.exp(l_sum / n)))
            for prefix in prefixes:
                print(prefix)
                print(' -', predict_rnn(prefix, pred_len, params))
        

#训练
pred_period, pred_len, prefixes = 50, 15, ['朝海', '分开']
clipping_theta = 0.01
batch_size = 256
train_and_predict_rnn(False, batch_size, clipping_theta, pred_period, pred_len, prefixes)











