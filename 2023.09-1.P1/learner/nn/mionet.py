"""
@author: jpzxshi
"""
import torch
from .module import Map
from .fnn import FNN

class MIONet(Map):
    '''Multiple-input operator network.
    '''
    def __init__(self, sizes, activation='relu', initializer='default'):
        super(MIONet, self).__init__()
        self.sizes = sizes
        self.activation = activation
        self.initializer = initializer

        self.ms = self.__init_modules()
        self.ps = self.__init_parameters()
    
    def forward(self, x):
        y = torch.stack([self.ms['Net{}'.format(i + 1)](x[i]) for i in range(len(self.sizes))])
        return torch.sum(torch.prod(y, dim=0), dim=-1, keepdim=True) + self.ps['bias']
    
    def __init_modules(self):
        modules = torch.nn.ModuleDict()
        for i in range(len(self.sizes)):
            modules['Net{}'.format(i + 1)] = FNN(self.sizes[i], self.activation, self.initializer)
        return modules
    
    def __init_parameters(self):
        parameters = torch.nn.ParameterDict()
        parameters['bias'] = torch.nn.Parameter(torch.zeros([1]))
        return parameters