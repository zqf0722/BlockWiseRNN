# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 13:24:41 2020

@author: zeng
"""

import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils import clip_grad_norm_
 
class Dictionary(object):
    '''
    构建word2id,id2word两个字典
    '''
    def __init__(self):
        self.word2idx = {} #字典 词到索引的映射
        self.idx2word = {} #字典  索引到词的映射
        self.idx = 0
    
    def add_word(self, word):
        if not word in self.word2idx: #如果词到索引的映射字典中 不包含该词 则添加
            self.word2idx[word] = self.idx 
            self.idx2word[self.idx] = word #同时创建索引到词的映射
            self.idx += 1
    
    def __len__(self):
        return len(self.word2idx) #词到索引映射的字典大小
 
 
class Corpus(object):
    '''
    基于训练语料，构建字典(word2id,id2word)
    '''
    def __init__(self):
        self.dictionary = Dictionary() #创建字典类对象
 
    def get_data(self, path, batch_size):
        # 添加词到字典
        with open(path, 'r') as f:#读取文件
            tokens = 0
            for line in f:  #遍历文件中的每一行
                words = line.split() + ['<eos>'] #以空格分隔 返回列表 并添加一个结束符<eos>
                tokens += len(words)
                for word in words: #将每个单词添加到字典中
                    self.dictionary.add_word(word)  
        
        # 对文件做Tokenize
        ids = torch.LongTensor(tokens)
        token = 0
        with open(path, 'r') as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1
        num_batches = ids.size(0) // batch_size
        ids = ids[:num_batches*batch_size]
        return ids.view(batch_size, -1).contiguous()
