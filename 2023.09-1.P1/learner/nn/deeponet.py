"""
@author: jpzxshi
"""
import torch
import torch.nn as nn
from .module import Map
from .fnn import FNN

class DeepONet(Map):
    '''Deep operator network.
    Input: ([batch size, branch_dim], [batch size, trunk_dim])
    Output: [batch size, 1]
    '''
    def __init__(self, branch_size, trunk_size, activation='relu', initializer='Glorot normal'):
        super(DeepONet, self).__init__()
        self.branch_size = branch_size
        self.trunk_size = trunk_size
        self.activation = activation
        self.initializer = initializer
        
        self.ms = self.__init_modules()
        self.ps = self.__init_params()
        
    def forward(self, x):
        x_branch, x_trunk = self.ms['Branch'](x[0]), self.ms['Trunk'](x[1])
        return torch.sum(x_branch * x_trunk, dim=-1, keepdim=True) + self.ps['bias']
        
    def __init_modules(self):
        modules = nn.ModuleDict()
        modules['Branch'] = FNN(self.branch_size, self.activation, self.initializer)
        modules['Trunk'] = FNN(self.trunk_size, self.activation, self.initializer)
        return modules
            
    def __init_params(self):
        params = nn.ParameterDict()
        params['bias'] = nn.Parameter(torch.zeros([1]))
        return params