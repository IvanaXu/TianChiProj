import math
import torch
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np




def cos_A2(input,beta):
    feature_g = input.t()
    norm2 = torch.norm(feature_g,p=2,dim=1).view(-1, 1)
    cos = beta*torch.div(torch.mm(feature_g, feature_g.t()), torch.mm(norm2, norm2.t()) + 1e-7)
    I = torch.eye(cos.shape[0]).cuda()     
    cos = F.softmax(cos,dim=1)
    cos = cos - (torch.triu(cos)-torch.triu(cos,diagonal=1))     
    mean = torch.mean(cos)
    cos[cos<mean]=0
    cos[cos>=mean]=1         
    cos = cos + I           
    rowsum = torch.sum(cos,dim=1)**(-0.5)
    D2 = torch.diag(rowsum)
    A2 = torch.mm(D2,cos)
    A2 = torch.mm(A2,D2)
    return A2


def cosM_A2(input,beta,A):
    feature_g = input.t()
    norm2 = torch.norm(feature_g,p=2,dim=1).view(-1, 1)
    cos = beta*torch.div(torch.mm(feature_g, feature_g.t()), torch.mm(norm2, norm2.t()) + 1e-7)
    I = torch.eye(cos.shape[0]).cuda()
    e = beta*torch.ones(cos.shape).cuda()
    A = torch.sigmoid(A)
    A = torch.triu(A)+torch.triu(A,diagonal=1).t()
    cos = torch.mm(0.5*cos+0.5*e,A)
        
    cos = cos - (torch.triu(cos)-torch.triu(cos,diagonal=1))         
    mean = torch.mean(cos)
    cos[cos<mean]=0
    cos[cos>=mean]=1         
    cos = cos + I           
    rowsum = torch.sum(cos,dim=1)**(-0.5)
    D2 = torch.diag(rowsum)
    A2 = torch.mm(D2,cos)
    A2 = torch.mm(A2,D2)
    return A2
    
def learn_A2 (A):
    A = torch.sigmoid(A)
    A = torch.triu(A,diagonal=1)
    I = torch.eye(A.shape[0]).cuda()
    A2 = A + A.t() +I             
                                   
    rowsum = torch.sum(A2,dim=1)**(-0.5)
    D2 = torch.diag(rowsum)
    A2 = torch.mm(D2,A2)
    A2 = torch.mm(A2,D2)
    return A2        
    

