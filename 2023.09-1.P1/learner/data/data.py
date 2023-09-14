"""
@author: jpzxshi
"""
import numpy as np
import torch
from ..utils import map_elementwise

class Data:
    '''Standard data format. 
    '''
    def __init__(self, X_train=None, y_train=None, X_test=None, y_test=None):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        
        self.__device = None
        self.__dtype = None
    
    def get_batch(self, batch_size):
        @map_elementwise
        def batch_mask(X, num):
            return np.random.choice(X.size(0), num, replace=False)
        @map_elementwise
        def batch(X, mask):
            return X[mask]
        mask = batch_mask(self.y_train, batch_size)
        return batch(self.X_train, mask), batch(self.y_train, mask)
    
    @property
    def device(self):
        return self.__device
    
    @property
    def dtype(self):
        return self.__dtype
    
    @device.setter    
    def device(self, d):
        if d == 'cpu':
            self.__to_cpu()
            self.__device = torch.device('cpu')
        elif d == 'gpu':
            self.__to_gpu()
            self.__device = torch.device('cuda')
        else:
            raise ValueError
    
    @dtype.setter     
    def dtype(self, d):
        if d == 'float':
            self.__to_float()
            self.__dtype = torch.float32
        elif d == 'double':
            self.__to_double()
            self.__dtype = torch.float64
        else:
            raise ValueError
    
    @property
    def dim(self):
        if isinstance(self.X_train, np.ndarray):
            return self.X_train.shape[-1]
        elif isinstance(self.X_train, torch.Tensor):
            return self.X_train.size(-1)
    
    @property
    def K(self):
        if isinstance(self.y_train, np.ndarray):
            return self.y_train.shape[-1]
        elif isinstance(self.y_train, torch.Tensor):
            return self.y_train.size(-1)
    
    @property
    def X_train_np(self):
        return Data.to_np(self.X_train)
    
    @property
    def y_train_np(self):
        return Data.to_np(self.y_train)
    
    @property
    def X_test_np(self):
        return Data.to_np(self.X_test)
    
    @property
    def y_test_np(self):
        return Data.to_np(self.y_test)
    
    @staticmethod
    @map_elementwise
    def to_np(d):
        if isinstance(d, np.ndarray) or d is None:
            return d
        elif isinstance(d, torch.Tensor):
            return d.cpu().detach().numpy()
        else:
            raise ValueError
    
    def __to_cpu(self):
        @map_elementwise
        def trans(d):
            if isinstance(d, np.ndarray):
                return torch.DoubleTensor(d)
            elif isinstance(d, torch.Tensor):
                return d.cpu()
        for d in ['X_train', 'y_train', 'X_test', 'y_test']:
            setattr(self, d, trans(getattr(self, d)))
    
    def __to_gpu(self):
        @map_elementwise
        def trans(d):
            if isinstance(d, np.ndarray):
                return torch.cuda.DoubleTensor(d)
            elif isinstance(d, torch.Tensor):
                return d.cuda()
        for d in ['X_train', 'y_train', 'X_test', 'y_test']:
            setattr(self, d, trans(getattr(self, d)))
    
    def __to_float(self):
        if self.device is None: 
            raise RuntimeError('device is not set')
        @map_elementwise
        def trans(d):
            if isinstance(d, torch.Tensor):
                return d.float()
        for d in ['X_train', 'y_train', 'X_test', 'y_test']:
            setattr(self, d, trans(getattr(self, d)))
    
    def __to_double(self):
        if self.device is None: 
            raise RuntimeError('device is not set')
        @map_elementwise
        def trans(d):
            if isinstance(d, torch.Tensor):
                return d.double()
        for d in ['X_train', 'y_train', 'X_test', 'y_test']:
            setattr(self, d, trans(getattr(self, d)))