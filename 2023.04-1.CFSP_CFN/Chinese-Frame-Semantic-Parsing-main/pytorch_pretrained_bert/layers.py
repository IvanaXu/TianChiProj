import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from pytorch_pretrained_bert.L2 import learn_A2, cos_A2, cosM_A2


class ADMM_Y(Module):
    def __init__(self,L,A2,p,lambda_1,lambda_2,n_iter=2,Type='mean'):
        super(ADMM_Y,self).__init__()
        self.p = p 
        self.L = L
        self.A2 = A2
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.n_iter = n_iter
        self.Type = Type
        
        self.I_2 = torch.eye(A2.shape[0]).cuda()
        self.L_2 = self.I_2-self.lambda_2*self.A2
    def ADMM_Y1(self,F,Y2,Z):
        T = F + self.p*Y2 + Z
        Y1 = 1./(1+self.p)*torch.mm(self.L,T)
        return Y1
    
    def ADMM_Y2(self,F,Y1,Z):
        T = F + self.p*Y1 - Z
        Y2 = 1./(1+self.p)*torch.mm(T,self.L_2)
        return Y2
        
    def ADMM_Z(self,Y1,Y2,Z):
        Z = Z + self.p*(Y2-Y1)
        return Z
    
    def forward(self,F,Y2,Z):
        for i in range(self.n_iter):
            Y1 = self.ADMM_Y1(F,Y2,Z)
            Y2 = self.ADMM_Y2(F,Y1,Z)
            Z = self.ADMM_Z(Y1,Y2,Z)
        if self.Type == 'y2':
            Y = Y2
        elif self.Type == 'y1':
            Y = Y1
        elif self.Type == 'mean':
            Y = 1/2*(Y1+Y2)
        return Y
#'''
class BiGCN_layer(Module):
    def __init__(self,ind,outd,p,lambda_1,lambda_2,bias=True,beta=True,A2='cos_A2',n_iter=2,Type='mean'):
        super(BiGCN_layer,self).__init__()
        self.ind = ind     #input dimension
        self.outd = outd   #output dimension
        self.p = p
        self.A2 = A2
        self.n_iter = n_iter
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.Type = Type
        
        if beta:
            self.beta = Parameter(torch.Tensor(1).uniform_(0, 1))
        else:
            self.beta = 1 
        self.weight1 = Parameter(torch.FloatTensor(ind,outd))
        self.A = Parameter(torch.FloatTensor(ind,ind))
        if bias:
            self.bias1 = Parameter(torch.FloatTensor(outd))
        else:
            self.register_parameter('bias1',None)
        self.reset_parameters()
    
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight1.size(1))
        self.weight1.data.uniform_(-stdv,stdv)
        self.A.data = torch.empty(self.A.shape).random_(2)
        if self.bias1 is not None:
            self.bias1.data.uniform_(-stdv,stdv)
   

    def forward(self,Y1,Y2,Z,L):
        if self.A2 == 'learn_A2':
            A2 = learn_A2(self.A)
        elif self.A2 == 'cos_A2':
            A2 = cos_A2(Y2,self.beta)
        elif self.A2 == 'cosM_A2':
            A2 = cosM_A2(Y2,self.beta,self.A)
        else:
            raise Exception("No such A2:",self.A2)
        admm = ADMM_Y(L,A2,self.p,self.lambda_1,self.lambda_2,self.n_iter,self.Type)
        Y = admm(Y1,Y2,Z)
        Y = torch.mm(Y,self.weight1)
        if self.bias1 is not None:
            return Y + self.bias1,A2
        else:
            return Y,A2

