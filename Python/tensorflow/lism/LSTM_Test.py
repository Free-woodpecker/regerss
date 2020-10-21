# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 19:24:06 2020
TensorFlow 2 中文文档测试学习
下载数据
@author: 爬上阁楼的鱼
"""
# geektutu.com
#import matplotlib as mpl
#import matplotlib.pyplot as plt

# 处理数据的库
#import numpy as np
#import sklearn
#import pandas as pd
# 系统库
#import os
#import sys
#import time
# TensorFlow的库
import tensorflow as tf
from tensorflow import keras


# IMDB：一个电影评分数据集，有两类，positive与negative
#下载
imdb = keras.datasets.imdb
vocab_size = 10000
index_from = 3
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(
                                                       num_words = vocab_size,  # 数据中词表的个数，根据词出现的频次，前10000个会被保留，其余当做特殊字符处理
                                                       index_from = index_from) # 词表的下标从3开始计算


print('ok')
print(train_data[0],train_labels[0])
print()
print(train_data.shape, train_labels.shape)   # 每个训练样本都是变长的
print()
print(len(train_data[0]), len(train_data[1])) # 比如第一个样本的长度是218，第二个样本的长度就是189
print()
print(test_data.shape, test_labels.shape) 


word_index = imdb.get_word_index() # 获取词表
#print(len(word_index)) # 打印词表长度
#print()

#print(list(word_index.items())[:50])  # 打印词表的前50个 key:value形式
#print()


"""
更改词表ID
因为我们读取词表的时候是从下标为3的时候开始计算的，所以此时要加3
使词表坐标偏移的目的是为了增加一些特殊字符
"""
word_index = {k:(v+3) for k, v in word_index.items()} 

word_index['<PAD>'] = 0      # padding时用来填充的字符
word_index['<START>'] = 1    # 每个句子开始之前的字符
word_index['<UNK>'] = 2      # 无法识别的字符
word_index['<END>'] = 3      # 每个句子结束时的字符

# id->word的索引
reverse_word_index = dict(
    [(value, key) for key, value in word_index.items()])

def decode_review(text_ids):
    return ' '.join(
        [reverse_word_index.get(word_id, "<UNK>") for word_id in text_ids]) # 没有找到的id默认用<UNK>代替
# 打印train_data[0]中ID对应的语句
decode_review(train_data[0])





max_length = 500 # 句子的长度，长度低于500的句子会被padding补齐，长度低于500的句子会被截断

"""
利用keras.preprocessing.sequence.pad_sequences对训练集与测试集数据进行补齐和截断
"""
# 处理训练集数据
train_data = keras.preprocessing.sequence.pad_sequences(
    train_data,                   # 要处理的数据
    value = word_index['<PAD>'],  # 要填充的值
    padding = 'post',             # padding的顺序：post指将padding放到句子的后面, pre指将padding放到句子的前面
    maxlen = max_length)          # 最大的长度

# 处理测试集数据
test_data = keras.preprocessing.sequence.pad_sequences(
    test_data,                    # 要处理的数据
    value = word_index['<PAD>'],  # 要填充的值
    padding = 'post',             # padding的顺序：post指将padding放到句子的后面, pre指将padding放到句子的前面
    maxlen = max_length)          # 最大的长度


#因为我们的样本是不等长的，每个评论的长度都不一样，所以我们需要对数据进行处理，规定一个样本长度，对于长度不足的样本进行补齐，对于长度超出的样本进行截断。
print(train_data[0])




# ####单向 RNN
# embedding_dim = 16  # 每个word embedding成一个长度为16的向量
# batch_size = 512    # 每个batch的长度

# """
# 单层单向的RNN
#     embedding层的作用
#     1. 定义一个矩阵 matrix: [vocab_size, embedding_dim] （[10000, 16]）
#     2. 对于每一个样本[1,2,3,4..],将其变为 max_length * embedding_dim维度的数据,即每一个词都变为长度为16的向量
#     3. 最后的数据为三维矩阵：batch_size * max_length * embedding_dim
# """
# model = keras.models.Sequential([
#     keras.layers.Embedding(vocab_size,                  # 词表的长度
#                            embedding_dim,               # embedding的长度
#                            input_length = max_length),  # 输入的长度
#     # units:输出空间维度
#     # return_sequences: 布尔值。是返回输出序列中的最后一个输出，还是全部序列。
#     keras.layers.SimpleRNN(units = 64, return_sequences = False),
#     # 全连接层
#     keras.layers.Dense(64, activation = 'relu'),
#     keras.layers.Dense(1, activation='sigmoid'),
# ])
# model.summary()



#####双向RNN
# embedding_dim = 16
# batch_size = 512

# model = keras.models.Sequential([
#     # 1. define matrix: [vocab_size, embedding_dim]
#     # 2. [1,2,3,4..], max_length * embedding_dim
#     # 3. batch_size * max_length * embedding_dim
#     keras.layers.Embedding(vocab_size, embedding_dim,
#                            input_length = max_length),
#     keras.layers.Bidirectional(
#         keras.layers.SimpleRNN(
#             units = 64, return_sequences = True)),
#     keras.layers.Dense(64, activation = 'relu'),
#     keras.layers.Dense(1, activation='sigmoid'),
# ])

# model.summary()
# model.compile(optimizer = 'adam',
#               loss = 'binary_crossentropy',
#               metrics = ['accuracy'])




"""
双向双层的LSTM
"""
embedding_dim = 16
batch_size = 512

#设定 LSTM model
model = keras.models.Sequential([
    # 1. define matrix: [vocab_size, embedding_dim]
    # 2. [1,2,3,4..], max_length * embedding_dim
    # 3. batch_size * max_length * embedding_dim
    keras.layers.Embedding(vocab_size, embedding_dim,
                           input_length = max_length),
    # units:输出空间维度
    # return_sequences: 布尔值。是返回输出序列中的最后一个输出，还是全部序列。
    keras.layers.Bidirectional(
        keras.layers.LSTM(
            units = 64, return_sequences = True)),
    keras.layers.Dense(64, activation = 'relu'),
    keras.layers.Dense(1, activation ='sigmoid'),
])

model.summary()
model.compile(optimizer = 'adam',
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])





# #train
# model.compile(optimizer = 'adam',
#                          loss = 'binary_crossentropy',
#                          metrics = ['accuracy'])
# history_single_rnn = model.fit(
#     train_data, train_labels,
#     epochs = 1,
#     batch_size = batch_size,
#     validation_split = 0.3)

# #save model
# #只填写路径即可
# model.save_weights('.\Model')
# model.save('.\LSTM.h5', save_format='tf')




