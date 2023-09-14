"""
@author: jpzxshi
"""
import abc
import torch
from ..utils import map_elementwise

class Module(torch.nn.Module):
    '''Standard module format.
    '''
    def __init__(self):
        super(Module, self).__init__()
        self.activation = None
        self.initializer = None
        
        self.__device = None
        self.__dtype = None
        
    @property
    def device(self):
        return self.__device
        
    @property
    def dtype(self):
        return self.__dtype

    @device.setter
    def device(self, d):
        if d == 'cpu':
            self.cpu()
            for module in self.modules():
                if isinstance(module, Module):
                    module.__device = torch.device('cpu')
        elif d == 'gpu':
            self.cuda()
            for module in self.modules():
                if isinstance(module, Module):
                    module.__device = torch.device('cuda')
        else:
            raise ValueError
    
    @dtype.setter
    def dtype(self, d):
        if d == 'float':
            self.to(torch.float32)
            for module in self.modules():
                if isinstance(module, Module):
                    module.__dtype = torch.float32
        elif d == 'double':
            self.to(torch.float64)
            for module in self.modules():
                if isinstance(module, Module):
                    module.__dtype = torch.float64
        else:
            raise ValueError

    @property
    def act(self):
        if callable(self.activation):
            return self.activation
        elif self.activation == 'sigmoid':
            return torch.sigmoid
        elif self.activation == 'relu':
            return torch.relu
        elif self.activation == 'tanh':
            return torch.tanh
        elif self.activation == 'elu':
            return torch.elu
        else:
            raise NotImplementedError

    @property
    def weight_init_(self):
        if callable(self.initializer):
            return self.initializer
        elif self.initializer == 'He normal':
            return torch.nn.init.kaiming_normal_
        elif self.initializer == 'He uniform':
            return torch.nn.init.kaiming_uniform_
        elif self.initializer == 'Glorot normal':
            return torch.nn.init.xavier_normal_
        elif self.initializer == 'Glorot uniform':
            return torch.nn.init.xavier_uniform_
        elif self.initializer == 'orthogonal':
            return torch.nn.init.orthogonal_
        elif self.initializer == 'default':
            if self.activation == 'relu':
                return torch.nn.init.kaiming_normal_
            elif self.activation == 'tanh':
                return torch.nn.init.orthogonal_
            else:
                return lambda x: None
        else:
            raise NotImplementedError
            
    @map_elementwise
    def _to_tensor(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=self.dtype, device=self.device)
        return x
            
class Map(Module):
    '''Structure-oriented neural network used as a general map based on designing architecture.
    '''
    def __init__(self):
        super(Map, self).__init__()
    
    def predict(self, x, returnnp=False):
        x = self._to_tensor(x)
        return self(x).cpu().detach().numpy() if returnnp else self(x)
    
class Algorithm(Module, abc.ABC):
    '''Loss-oriented neural network used as an algorithm based on designing loss.
    '''
    def __init__(self):
        super(Algorithm, self).__init__()
        
    #@final
    def forward(self, x):
        return x
    
    @abc.abstractmethod
    def criterion(self, X, y):
        pass
    
    @abc.abstractmethod
    def predict(self):
        pass