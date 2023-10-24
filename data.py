import os 
import numpy as np
import pandas as pd
from scipy.stats import ortho_group

import glob
import random

look_back1=20
look_back2=50

def gen_proxy():    
    # generate a unitary matrix
    m = ortho_group.rvs(dim=32)
    x=m[0:8]
    return(x.reshape(32,1))

def gen_real_proxy(Y1):
    # generate a vector of same dimension as Y1 (which is 32) and it
    # should a correlation of at least 0.8 (>0.8) with 
    # Y1.
    

def lstm_data(f):
    # read dataset
    df = pd.read_csv(f,encoding='utf-16',usecols=list(range(0,80))) # 80D BNF Features
    dt = df.astype(np.float32)
    X=np.array(dt)
    
    Xdata1=[]
    Xdata2=[] 
    Ydata1 =[]
      
    mu = X.mean(axis=0)
    std = X.std(axis=0)
    np.place(std, std == 0, 1) 
    X = (X - mu) / std 
    f1 = os.path.splitext(f)[0]     
    
    lang = f1[63:66]  
    
    # generate proxy vectors
    o = gen_proxy()
                
    if(lang == 'asm'):
        Y1 = o[0]
        
    elif(lang == 'ben'):
        Y1 = o[1]              
    
    elif(lang == 'kan'):
        Y1 = o[2]
        
    elif(lang == 'hin'):
        Y1 = o[3]
        
    elif(lang == 'tel'):
        Y1 = o[4]
        
    elif(lang == 'odi'):
        Y1 = o[5]
    
    elif(lang == 'guj'):
        Y1 = o[6]
        
    elif(lang == 'mal'):
        Y1 = o[7]        
    
    Y2=np.array([Y1]) 
     
    # High resolution low context
    for i in range(0,len(X)-look_back1,1):            
        a=X[i:(i+look_back1),:]        
        Xdata1.append(a)
    Xdata1=np.array(Xdata1)

    # Low resolution long context
    for i in range(0,len(X)-look_back2,2):            
        b=X[i:(i+look_back2):3,:]        
        Xdata2.append(b)
    Xdata2=np.array(Xdata2)

    # form datasets
    Xdata1 = torch.from_numpy(Xdata1).float()
    Xdata2 = torch.from_numpy(Xdata2).float()
    Ydata1 = torch.from_numpy(Y2).long()
    Yproxy1 = torch.from_numpy(gen_real_proxy(Y1)).long()
    
    return Xdata1,Xdata2,Ydata1,Yproxy1