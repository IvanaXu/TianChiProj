"""
@author: jpzxshi
"""
import torch.nn as nn
from .module import Map

class FNN(Map):
    '''Fully-connected neural network.
    Note that
    len(size) >= 2,
    [..., N1, -N2, ...] denotes a linear layer from dim N1 to N2 without bias,
    [..., N, 0] denotes an identity map (as output linear layer).
    '''
    def __init__(self, size, activation='relu', initializer='default'):
        super(FNN, self).__init__()
        self.size = size
        self.activation = activation
        self.initializer = initializer
        
        self.ms = self.__init_modules()
        self.__initialize()
        
    def forward(self, x):
        for i in range(1, len(self.size) - 1):
            x = self.act(self.ms['LinM{}'.format(i)](x))
        return self.ms['LinM{}'.format(len(self.size) - 1)](x) if self.size[-1] != 0 else x
    
    def __init_modules(self):
        modules = nn.ModuleDict()
        for i in range(1, len(self.size)):
            if self.size[i] != 0:
                bias = True if self.size[i] > 0 else False
                modules['LinM{}'.format(i)] = nn.Linear(abs(self.size[i - 1]), abs(self.size[i]), bias)
        return modules
    
    def __initialize(self):
        for i in range(1, len(self.size)):
            if self.size[i] != 0: 
                self.weight_init_(self.ms['LinM{}'.format(i)].weight)
                if self.size[i] > 0:
                    nn.init.constant_(self.ms['LinM{}'.format(i)].bias, 0)