# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 16:15:06 2020

@author: zeng
"""

import torch
import torch.nn as nn
import torch.nn.functional as f
#import time

class FSMNLM(nn.Module):
    def __init__(self, front, vocab_size, embed_size, hidden_size, seq_length, batch_size, dropout):
        super(FSMNLM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.hidden1 = nn.Sequential(nn.Linear(embed_size, hidden_size), nn.ReLU(True))

        self.memory_block = nn.Linear(front*seq_length, seq_length)
        self.hidden2 = nn.Sequential(nn.Linear(2*hidden_size, hidden_size), nn.ReLU(True))
        self.linear = nn.Linear(hidden_size,vocab_size)
        self.ntokens = vocab_size
        self.nhid = hidden_size
        self.order = front
        self.seq = seq_length
        self.batch = batch_size
        self.drop = nn.Dropout(dropout)

        
    def forward(self, x, h):
        emb = self.drop(self.embed(x))
        h1 = self.drop(self.hidden1(emb))
        h_in = torch.cat((h,h1),1)
        h_out = h_in[:,self.seq:self.order*self.seq,:]
        l2_input = h_in.transpose(1, 2).contiguous()
        h_memory = self.memory_block(l2_input)    
        l3_input = h_memory.transpose(1,2).contiguous()
        l3_input = torch.cat((h1,l3_input),2)
        h2_output = self.drop(self.hidden2(l3_input))
        outputs = self.linear(h2_output)
        out = outputs.view(-1,self.ntokens)
        

        
        return f.log_softmax(out, dim=1), h_out