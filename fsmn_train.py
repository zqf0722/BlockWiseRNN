# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 20:02:50 2020

@author: zeng
"""



import argparse
import pre
import fsmn_baseline as fsmn
import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils import clip_grad_norm_

parser = argparse.ArgumentParser(description='PyTorch PTB LSTM Language Model')

parser.add_argument('--data', type=str, default='./data/ptb.train.txt',
                    help='location of the data corpus')
parser.add_argument('--test', type=str, default='./data/ptb.test.txt',
                    help='location of the test corpus')
parser.add_argument('--valid', type=str, default='./data/ptb.valid.txt',
                    help='location of the test corpus')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--front', type=int, default=20,
                    help='size of word to consider')
parser.add_argument('--nhid', type=int, default=400,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=0.4,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=6,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=2,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.25,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--save', type=str, default='model/model.pt',
                    help='path to save the final model')
parser.add_argument('--eval_size', type=int, default=100, metavar='N',
                    help='evaluate batch size')

args = parser.parse_args()

ori_lr = args.lr

# 优先使用GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载数据集
corpus = pre.Corpus()
ids = corpus.get_data(args.data, args.batch_size)
test_data = corpus.get_data(args.test, args.eval_size)
valid_data = corpus.get_data(args.valid, args.eval_size)
vocab_size = len(corpus.dictionary)
num_batches = ids.size(1) // args.bptt
num_batches2 = test_data.size(1) // args.bptt
#print(ids.shape)

#引用FSMN模型
model = fsmn.FSMNLM(args.front, vocab_size, args.emsize, args.nhid, args.bptt,args.batch_size, args.dropout).to(device)

# 损失构建与优化
criterion = nn.CrossEntropyLoss()

#训练
# 反向传播过程“截断”(不复制gradient)
def detach(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

epoch = 0
flag = 0
count = 0 
j = 0
# 训练模型


for epoch in range(args.epochs):
    #初始化之前的隐藏层输出
    h = torch.zeros([args.batch_size, (args.front-1)*args.bptt, args.nhid]).to(device)
    model.train()
    for i in range(0, ids.size(1) - args.bptt, args.bptt):
        # 获取一个mini batch的输入和输出(标签)
        inputs = ids[:,i:i+args.bptt].to(device)
        targets = ids[:,(i+1):(i+1)+args.bptt].to(device) # 输出相对输入错一位，往后顺延一个单词
        
        
        # 前向运算
        h=detach(h)
        outputs,h = model(inputs,h)
        
        loss = criterion(outputs,targets.reshape(-1))
        
        # 反向传播与优化
        model.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(),args.clip)
        #梯度下降
        #since = time.time()
        optimizer = torch.optim.SGD(model.parameters(), lr = args.lr, weight_decay = 0.00004, momentum = 0.9)
        '''
        for p in model.parameters():
            p.data.add_(-args.lr, p.grad.data)
        '''
        optimizer.step()
        #print('grad time cost{:.0f}' .format((time.time()-since)%60))
        step = (i+1) // args.bptt
        if step % 200 == 0:
            print ('epoch [{}], Steps[{}/{}], Loss: {:.4f}, Perplexity: {:5.2f}'
                   .format(epoch+1,  step, num_batches, loss.item(), np.exp(loss.item())))
    torch.save(model,args.save)
    model.eval()   
    with torch.no_grad():
        loss_overall=0
        h_valid = torch.zeros([args.eval_size, (args.front-1)*args.bptt, args.nhid]).to(device)
        for i in range(0, valid_data.size(1) - args.bptt, args.bptt):
            inputs2 = valid_data[:,i:i+args.bptt].to(device)
            targets2 = valid_data[:,(i+1):(i+1)+args.bptt].to(device) # 输出相对输入错一位，往后顺延一个单词
            h_valid = detach(h_valid)
            outputs2,h_valid = model(inputs2,h_valid)
            
                
                # 计算perplexity
            
            loss2 = criterion(outputs2,targets2.reshape(-1))
            loss_overall=loss2.item()+loss_overall
        loss_overall=loss_overall/num_batches2
        '''
        if j == 0:
            best_loss = loss_overall
        if np.exp(best_loss) - np.exp(loss_overall) >= 1:
               if flag == 0:
                   if best_loss >= loss_overall:
                       best_loss = loss_overall
                       torch.save(model, args.save)
               else:
                   if best_loss >= loss_overall:
                       torch.save(model, args.save)
                       count += 1
                       args.lr = args.lr/2                  
        else:
            if flag == 1:
                count += 1
                args.lr = args.lr/2
            else:
                flag = 1
        '''        
        print('valid Loss{:5.2f}, Perplexity:{:8.2f}' .format(loss_overall,np.exp(loss_overall)))
#测试
model = torch.load(args.save)
model.eval()



#总的损失函数
loss_overall=0



with torch.no_grad():

        h_test = torch.zeros([args.eval_size, (args.front-1)*args.bptt, args.nhid]).to(device)
        for i in range(0, test_data.size(1) - args.bptt, args.bptt):
            inputs2 = test_data[:,i:i+args.bptt].to(device)
            targets2 = test_data[:,(i+1):(i+1)+args.bptt].to(device) # 输出相对输入错一位，往后顺延一个单词
            h_test = detach(h_test)
            outputs2,h_test = model(inputs2,h_test)
            
            
            # 计算perplexity
            
            loss2 = criterion(outputs2,targets2.reshape(-1))
            loss_overall=loss2.item()+loss_overall
        loss_overall=loss_overall/num_batches2
            
        print('test Cultimate Loss{:5.2f},Cultimate Perplexity:{:8.2f}' .format(loss_overall,np.exp(loss_overall)))
print('end of training')

#保存模型
