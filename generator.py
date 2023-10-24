import torch
import torch.nn as nn
import torch.nn.functional as F

import os 
import numpy as np
import pandas as pd

import glob
import random

from torch.autograd import Variable
from torch.autograd import Function
from torch import optim

# create the generator
class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.lstm1 = nn.LSTM(80, 256,bidirectional=True)
        self.lstm2 = nn.LSTM(2*256, 32,bidirectional=True)
               
        self.fc_ha=nn.Linear(2*32,100) 
        self.fc_1= nn.Linear(100,1)
        self.sftmax = nn.Softmax(dim=1)

    def forward(self, x):
        x1, _ = self.lstm1(x) 
        x2, _ = self.lstm2(x1)
        ht = x2[-1]
        ht=torch.unsqueeze(ht, 0)        
        ha= torch.tanh(self.fc_ha(ht))
        alp= self.fc_1(ha) 
        al= self.sftmax(alp) 
        
        T=list(ht.shape)[1]  
        batch_size=list(ht.shape)[0]
        D=list(ht.shape)[2]
        
        # Self-attention on LID-seq-senones to get utterance-level embedding (e1/e2)      
        c=torch.bmm(al.view(batch_size, 1, T),ht.view(batch_size,T,D))  
        c = torch.squeeze(c,0)        
        return (c)