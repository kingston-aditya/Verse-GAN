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

# create the discriminator
class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        
        self.att1=nn.Linear(2*32,100) 
        self.att2= nn.Linear(100,1)
        
        self.bsftmax = nn.Softmax(dim=1)

        self.lang_classifier= nn.Sequential()
        self.lang_classifier.add_module('fc1',nn.Linear(2*32,48,bias=True))
        self.lang_classifier.add_module('fc2',nn.Linear(48,1,bias=True))
        
        
    def forward(self, ht_u):
#         u1 = self.model1(x1)
#         u2 = self.model2(x2)        
#         ht_u = torch.cat((u1,u2), dim=0)  
        ht_u = torch.unsqueeze(ht_u, 0) 
        ha_u = torch.tanh(self.att1(ht_u))
        alp = torch.tanh(self.att2(ha_u))
        al= self.bsftmax(alp)
        Tb = list(ht_u.shape)[1] 
        batch_size = list(ht_u.shape)[0]
        D = list(ht_u.shape)[2]
        
        # Self-attention combination of e1 and e2 to get u-vec
        u_vec = torch.bmm(al.view(batch_size, 1, Tb),ht_u.view(batch_size,Tb,D)) 
        u_vec = torch.squeeze(u_vec,0)
        
        # Output layer 
        rorf = self.lang_classifier(u_vec)       
        
        return (rorf)