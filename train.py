import torch
import torchvision.transforms as transforms
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
from pytorch_metric_learning import losses

# calling user defined functions
from discriminator import discriminator
from generator import generator
from data import lstm_data

# initialize custom loss functions 
class WSM_loss(nn.Module):
    def __init__(self):
        super(WSM_loss, self).__init__()
        
    def forward(self,e1,e2)
        loss_wssl = nn.CosineSimilarity()
        m = nn.Sigmoid()
        l1 = loss_wssl(e1,e2)
        return(m(l1))
    
class MSProduct(nn.Module):
    def __init__(self, margin):
        super(MSProduct, self).__init__()
        self.margin = margin
    
    def forward(self,inp,label):
        m = nn.LogSoftmax(dim=1)
        cosdist = nn.CosineSimilarity(F.normalize(inp), F.normalize(label))
        cosm = torch.acos(cosdist)+self.margin
        cosdist = 1*torch.cos(cosm)
        k = m(cosdist)
        return(k)
        
    
if __name__ == '__main__':
    # initialize training
    genr1 = generator()
    genr2 = generator()

    genr1.cuda()
    genr2.cuda()

    disc = discriminator(genr1,genr2)
    disc.cuda()
    
    # optimizers
    optim_disc = optim.SGD(disc.parameters(),lr = 0.01, momentum= 0.9)
    optim_genr1 = optim.SGD(genr1.parameters(),lr = 0.01, momentum= 0.9)
    optim_genr2 = optim.SGD(genr2.parameters(),lr = 0.01, momentum= 0.9)

    # initialising epochs
    n_epoch = 20
    manual_seed = random.randint(1,10000)
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)

    files_list=[]
    
    # Train files in csv format
    folders = glob.glob('/u/home/a/asarkar/scratch/VerseGAN/bnf_embeds/*')  
    for folder in folders:
        for f in glob.glob(folder+'/*.csv'):
            files_list.append(f)

    l = len(files_list)
    random.shuffle(files_list)
    print('Total Training files: ',l)
    
    # finished initialization
    print('#'*30)
    
    genr1.train()
    genr2.train()
    disc.train()
    
    # initialsing loss
    msl = MSProduct(0.2)
    wsml = WSM_loss()
    
    
    for e in range(n_epoch):
        cost = 0.
        random.shuffle(files_list)
        
        # number of files completed in the epoch
        i=0  
        for fn in files_list:    
            #print(fn)
            df = pd.read_csv(fn,encoding='utf-16',usecols=list(range(0,80)))
            data = df.astype(np.float32)
            X = np.array(data) 
            N,D=X.shape

            if N>look_back2:
                
                #disc.zero_grad()
                #gene1.zero_grad()
                #gene2.zero_grad()
                
                XX1,XX2,YY1,YP1 = lstm_data(fn)
                XNP=np.array(XX1)
                if(np.isnan(np.sum(XNP))):
                    continue

                XNP=np.array(XX2)
                if(np.isnan(np.sum(XNP))):
                    continue

                i = i+1
                XX1 = np.swapaxes(XX1,0,1)
                XX2 = np.swapaxes(XX2,0,1)
                X1 = Variable(XX1,requires_grad=False).cuda()
                Y1 = Variable(YY1,requires_grad=False).cuda()
                X2 = Variable(XX2,requires_grad=False).cuda()
                YP1 = Variable(YP1,requires_grad=False).cuda()
                
                # initialise real vector
                real = torch.cat((YP1,YP1),dim=0)
                
                # initialise fake vector
                fake1 = genr1.forward(X1)
                fake2 = genr2.forward(X2)
                fake = torch.cat((fake1,fake2), dim=0)

                # training discriminator
                disc_real = disc.forward(real)
                lossD_real = msl.forward(disc_real,torch.ones_like(disc_real))
                disc_fake = disc.forward(fake)
                lossD_fake = msl.forward(disc_fake,torch.zeros_like(disc_fake))
                lossD = 0.5*(lossD_real+lossD_fake)
                disc.zero_grad()
                lossD.backward(retain_graph=True)
                optim_disc.step()
                
                # training generator
                output = disc.foward(fake)
                lossG = msl.forward(output, torch.ones_like(output)) + 0.4*wsml.forward(fake1, fake2)
                genr1.zero_grad()
                genr2.zero_grad()
                lossG.backward()
                optim_gen.step()

                print("VerseGAN:  epoch "+str(e+1)+" completed files  "+str(i)+"/"+str(l)+" Generator Loss= %.3f"%(lossG/i))
                print("VerseGAN:  epoch "+str(e+1)+" completed files  "+str(i)+"/"+str(l)+" Discriminator Loss= %.3f"%(lossD/i))

        # Save the weights of generator after every epoch
        path = "/u/home/a/asarkar/scratch/VerseGAN/"+str(e+1)+".pth" 
        torch.save(genr1.state_dict(),os.path.join(path,"genr1"))
        torch.save(genr2.state_dict(),os.path.join(path,"genr2"))
        torch.save(disc.state_dict(),os.path.join(path,"disc"))