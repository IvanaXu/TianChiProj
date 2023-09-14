"""
@author: jpzxshi
"""
import torch
from .module import Module, Map
from .fnn import FNN

class AdditiveCouplingLayer(Module):
    def __init__(self, D, d, layers, width, activation, initializer, mode):
        super(AdditiveCouplingLayer, self).__init__()
        self.D = D
        self.d = d
        self.layers = layers
        self.width = width
        self.activation = activation
        self.initializer = initializer
        self.mode = mode
        
        self.ms = self.__init_modules()
        
    def forward(self, x1x2):
        x1, x2 = x1x2
        if self.mode == 'up':
            return x1 + self.ms['m'](x2), x2
        elif self.mode == 'low':
            return x1, x2 + self.ms['m'](x1)
        else:
            raise ValueError
            
    def inverse(self, y1y2):
        y1, y2 = y1y2
        if self.mode == 'up':
            return y1 - self.ms['m'](y2), y2
        elif self.mode == 'low':
            return y1, y2 - self.ms['m'](y1)
        else:
            raise ValueError
            
    def __init_modules(self):
        modules = torch.nn.ModuleDict()
        din, dout = (self.d, self.D - self.d) if self.mode == 'low' else (self.D - self.d, self.d)
        modules['m'] = FNN([din] + [self.width] * (self.layers - 1) + [dout], self.activation, self.initializer)
        return modules
    
class CouplingLayer(Module):
    def __init__(self, D, d, layers, width, activation, initializer, mode):
        super(CouplingLayer, self).__init__()
        self.D = D
        self.d = d
        self.layers = layers
        self.width = width
        self.activation = activation
        self.initializer = initializer
        self.mode = mode
        
        self.ms = self.__init_modules()
        
    def forward(self, x1x2):
        x1, x2 = x1x2
        if self.mode == 'up':
            return x1 * torch.exp(self.ms['s'](x2)) + self.ms['t'](x2), x2
        elif self.mode == 'low':
            return x1, x2 * torch.exp(self.ms['s'](x1)) + self.ms['t'](x1)
        else:
            raise ValueError
            
    def inverse(self, y1y2):
        y1, y2 = y1y2
        if self.mode == 'up':
            return (y1 - self.ms['t'](y2)) * torch.exp(-self.ms['s'](y2)), y2
        elif self.mode == 'low':
            return y1, (y2 - self.ms['t'](y1)) * torch.exp(-self.ms['s'](y1))
        else:
            raise ValueError
            
    def __init_modules(self):
        modules = torch.nn.ModuleDict()
        din, dout = (self.d, self.D - self.d) if self.mode == 'low' else (self.D - self.d, self.d)
        modules['s'] = FNN([din] + [self.width] * (self.layers - 1) + [dout], self.activation, self.initializer)
        modules['t'] = FNN([din] + [self.width] * (self.layers - 1) + [dout], self.activation, self.initializer)
        return modules
    
class INN(Map):
    '''Invertible neural network. (NICE and real NVP)
    '''
    def __init__(self, D, d, layers=3, sublayers=3, subwidth=20, activation='sigmoid', initializer='default', volume_preserving=False):
        super(INN, self).__init__()
        self.D = D
        self.d = d
        self.layers = layers
        self.sublayers = sublayers
        self.subwidth = subwidth
        self.activation = activation
        self.initializer = initializer
        self.volume_preserving = volume_preserving
        
        self.ms = self.__init_modules()
        
    def forward(self, x):
        x = x[..., :self.d], x[..., self.d:]
        for i in range(self.layers):
            x = self.ms['M{}'.format(i+1)](x)
        return torch.cat(x, -1)
    
    def inverse(self, y):
        y = y[..., :self.d], y[..., self.d:]
        for i in reversed(range(self.layers)):
            y = self.ms['M{}'.format(i+1)].inverse(y)
        return torch.cat(y, -1)
    
    def __init_modules(self):
        modules = torch.nn.ModuleDict()
        for i in range(self.layers):
            mode = 'up' if i % 2 == 0 else 'low'
            if self.volume_preserving:
                modules['M{}'.format(i+1)] = AdditiveCouplingLayer(self.D, self.d, self.sublayers, self.subwidth, self.activation, self.initializer, mode)
            else:
                modules['M{}'.format(i+1)] = CouplingLayer(self.D, self.d, self.sublayers, self.subwidth, self.activation, self.initializer, mode)
        return modules