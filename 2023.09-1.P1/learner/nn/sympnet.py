"""
@author: jpzxshi
"""
import torch
import torch.nn as nn
from .module import Module, Map

class LinearModule(Module):
    '''Linear symplectic module.
    '''
    def __init__(self, dim, layers):
        super(LinearModule, self).__init__()
        self.dim = dim
        self.layers = layers
        
        self.ps = self.__init_params()
        
    def forward(self, pqh):
        p, q, h = pqh
        for i in range(self.layers):
            S = self.ps['S{}'.format(i + 1)]
            if i % 2 == 0:
                p = p + q @ (S + S.t()) * h
            else:
                q = p @ (S + S.t()) * h + q
        return p + self.ps['bp'] * h, q + self.ps['bq'] * h
    
    def __init_params(self):
        '''Si is distributed N(0, 0.01), and b is set to zero.
        '''
        d = self.dim // 2
        params = nn.ParameterDict()
        for i in range(self.layers):
            params['S{}'.format(i + 1)] = nn.Parameter((torch.randn([d, d]) * 0.01).requires_grad_(True))
        params['bp'] = nn.Parameter(torch.zeros([d]).requires_grad_(True))
        params['bq'] = nn.Parameter(torch.zeros([d]).requires_grad_(True))
        return params
        
class ActivationModule(Module):
    '''Activation symplectic module.
    '''
    def __init__(self, dim, activation, mode):
        super(ActivationModule, self).__init__()
        self.dim = dim
        self.activation = activation
        self.mode = mode
        
        self.ps = self.__init_params()
        
    def forward(self, pqh):
        p, q, h = pqh
        if self.mode == 'up':
            return p + self.act(q) * self.ps['a'] * h, q
        elif self.mode == 'low':
            return p, self.act(p) * self.ps['a'] * h + q
        else:
            raise ValueError
            
    def __init_params(self):
        d = self.dim // 2
        params = nn.ParameterDict()
        params['a'] = nn.Parameter((torch.randn([d]) * 0.01).requires_grad_(True))
        return params
            
class GradientModule(Module):
    '''Gradient symplectic module.
    '''
    def __init__(self, dim, width, activation, mode):
        super(GradientModule, self).__init__()
        self.dim = dim
        self.width = width
        self.activation = activation
        self.mode = mode
        
        self.ps = self.__init_params()
        
    def forward(self, pqh):
        p, q, h = pqh
        if self.mode == 'up':
            gradH = (self.act(q @ self.ps['K'] + self.ps['b']) * self.ps['a']) @ self.ps['K'].t()
            return p + gradH * h, q
        elif self.mode == 'low':
            gradH = (self.act(p @ self.ps['K'] + self.ps['b']) * self.ps['a']) @ self.ps['K'].t()
            return p, gradH * h + q
        else:
            raise ValueError
            
    def __init_params(self):
        d = self.dim // 2
        params = nn.ParameterDict()
        params['K'] = nn.Parameter((torch.randn([d, self.width]) * 0.01).requires_grad_(True))
        params['a'] = nn.Parameter((torch.randn([self.width]) * 0.01).requires_grad_(True))
        params['b'] = nn.Parameter(torch.zeros([self.width]).requires_grad_(True))
        return params
    
class ExtendedModule(Module):
    '''Extended symplectic module.
    '''
    def __init__(self, dim, latent_dim, width, activation, mode):
        super(ExtendedModule, self).__init__()
        self.dim = dim
        self.latent_dim = latent_dim
        self.width = width
        self.activation = activation
        self.mode = mode
        
        self.ps = self.__init_params()
        
    def forward(self, pqch):
        p, q, c, h = pqch
        if self.mode == 'up':
            gradH = (self.act(q @ self.ps['K1'] + c @ self.ps['K2'] + self.ps['b']) * self.ps['a']) @ self.ps['K1'].t()
            return p + gradH * h, q, c
        elif self.mode == 'low':
            gradH = (self.act(p @ self.ps['K1'] + c @ self.ps['K2'] + self.ps['b']) * self.ps['a']) @ self.ps['K1'].t()
            return p, gradH * h + q, c
        else:
            raise ValueError
            
    def __init_params(self):
        d, dc = self.latent_dim // 2, self.dim - self.latent_dim
        params = nn.ParameterDict()
        params['K1'] = nn.Parameter((torch.randn([d, self.width]) * 0.01).requires_grad_(True))
        params['K2'] = nn.Parameter((torch.randn([dc, self.width]) * 0.01).requires_grad_(True))
        params['a'] = nn.Parameter((torch.randn([self.width]) * 0.01).requires_grad_(True))
        params['b'] = nn.Parameter(torch.zeros([self.width]).requires_grad_(True))
        return params
    
class SympNet(Map):
    def __init__(self):
        super(SympNet, self).__init__()
        self.dim = None
        
    def predict(self, xh, steps=1, keepinitx=False, returnnp=False):
        xh = self._to_tensor(xh)
        if type(xh) in (list, tuple):
            size = len(xh[0].size())
            x0, h = xh[0], xh[1] 
            pred = [x0]
            for _ in range(steps):
                pred.append(self((pred[-1], h)))
        else:
            size = len(xh.size())
            pred = [xh]
            for _ in range(steps):
                pred.append(self(pred[-1]))
        if keepinitx:
            steps = steps + 1
        else:
            pred = pred[1:]
        res = torch.cat(pred, dim=-1)
        if steps > 1:
            res = res.view([-1, steps, self.dim][2 - size:])
        return res.cpu().detach().numpy() if returnnp else res

class LASympNet(SympNet):
    '''LA-SympNet.
    Input: [num, dim] or ([num, dim], [num, 1])
    Output: [num, dim]
    '''
    def __init__(self, dim, layers=3, sublayers=2, activation='sigmoid'):
        super(LASympNet, self).__init__()
        self.dim = dim
        self.layers = layers
        self.sublayers = sublayers
        self.activation = activation
        
        self.ms = self.__init_modules()
        
    def forward(self, pqh):
        d = self.dim // 2
        if type(pqh) in (list, tuple):
            p, q, h = pqh[0][..., :d], pqh[0][..., d:], pqh[1]
        else:
            p, q, h = pqh[..., :d], pqh[..., d:], torch.ones_like(pqh[..., -1:])
        for i in range(self.layers - 1):
            LinM = self.ms['LinM{}'.format(i + 1)]
            ActM = self.ms['ActM{}'.format(i + 1)]
            p, q = ActM([*LinM([p, q, h]), h])
        return torch.cat(self.ms['LinMout']([p, q, h]), dim=-1)
    
    def __init_modules(self):
        modules = nn.ModuleDict()
        for i in range(self.layers - 1):
            modules['LinM{}'.format(i + 1)] = LinearModule(self.dim, self.sublayers)
            mode = 'up' if i % 2 == 0 else 'low'
            modules['ActM{}'.format(i + 1)] = ActivationModule(self.dim, self.activation, mode)
        modules['LinMout'] = LinearModule(self.dim, self.sublayers)
        return modules
            
class GSympNet(SympNet):
    '''G-SympNet.
    Input: [num, dim] or ([num, dim], [num, 1])
    Output: [num, dim]
    '''
    def __init__(self, dim, layers=3, width=20, activation='sigmoid'):
        super(GSympNet, self).__init__()
        self.dim = dim
        self.layers = layers
        self.width = width
        self.activation = activation
        
        self.ms = self.__init_modules()
        
    def forward(self, pqh):
        d = self.dim // 2
        if type(pqh) in (list, tuple):
            p, q, h = pqh[0][..., :d], pqh[0][..., d:], pqh[1]
        else:
            p, q, h = pqh[..., :d], pqh[..., d:], torch.ones_like(pqh[..., -1:])
        for i in range(self.layers):
            GradM = self.ms['GradM{}'.format(i + 1)]
            p, q = GradM([p, q, h])
        return torch.cat([p, q], dim=-1)
    
    def __init_modules(self):
        modules = nn.ModuleDict()
        for i in range(self.layers):
            mode = 'up' if i % 2 == 0 else 'low'
            modules['GradM{}'.format(i + 1)] = GradientModule(self.dim, self.width, self.activation, mode)
        return modules
    
class ESympNet(SympNet):
    '''E-SympNet.
    Input: [num, dim] or ([num, dim], [num, 1])
    Output: [num, dim]
    '''
    def __init__(self, dim, latent_dim, layers=3, width=20, activation='sigmoid'):
        super(ESympNet, self).__init__()
        self.dim = dim
        self.latent_dim = latent_dim
        self.layers = layers
        self.width = width
        self.activation = activation
        
        self.ms = self.__init_modules()
        
    def forward(self, pqch):
        d = self.latent_dim // 2
        if type(pqch) in (list, tuple):
            p, q, c, h = pqch[0][..., :d], pqch[0][..., d:2*d], pqch[0][..., 2*d:-1], pqch[1]
        else:
            p, q, c, h = pqch[..., :d], pqch[..., d:2*d], pqch[..., 2*d:], torch.ones_like(pqch[..., -1:])
        for i in range(self.layers):
            ExtM = self.ms['ExtM{}'.format(i + 1)]
            p, q, c = ExtM([p, q, c, h])
        return torch.cat([p, q, c], dim=-1)
    
    def __init_modules(self):
        modules = nn.ModuleDict()
        for i in range(self.layers):
            mode = 'up' if i % 2 == 0 else 'low'
            modules['ExtM{}'.format(i + 1)] = ExtendedModule(self.dim, self.latent_dim, self.width, self.activation, mode)
        return modules