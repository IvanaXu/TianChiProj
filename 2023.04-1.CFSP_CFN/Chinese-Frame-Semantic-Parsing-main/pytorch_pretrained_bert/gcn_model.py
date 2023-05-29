import math
import torch
import torch.nn as nn 
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.autograd import Variable
import numpy as np
from .layers import BiGCN_layer


    
class BiGCN(nn.Module):
    def __init__(self,num_feature,num_hidden,num_class,p,lambda_1,lambda_2,dropout,bias=True,beta=True,A2='cos_A2',n_iter=2,Type='mean'):
        super(BiGCN,self).__init__()
        
        self.p = p
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.Type = Type
        self.num_feature = num_feature
        self.num_hidden = num_hidden
        self.gc1 = BiGCN_layer(num_feature,num_hidden,p,lambda_1,lambda_2,bias,beta,A2,n_iter,Type) 
        self.gc2 = BiGCN_layer(num_hidden,num_class,p,lambda_1,lambda_2,bias,beta,A2,n_iter,Type)
        self.dropout = dropout
        self.reg_params = list(self.gc1.parameters())
        self.non_reg_params = list(self.gc2.parameters())
        
    def forward(self, x, L):
        A = []
        Z1 = torch.zeros(x.shape[0], self.num_feature).cuda()
        X = F.relu(self.gc1(x, x, Z1, L)[0])
        A.append(self.gc1(x, x, Z1, L)[1])
        X = F.dropout(X, self.dropout, training=self.training)
        Z2 = torch.zeros(x.shape[0], self.num_hidden).cuda()
        X, _ = self.gc2(X, X, Z2, L)
        return X
       
