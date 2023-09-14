"""
@author: jpzxshi
"""
import torch
from .module import Map
from .fnn import FNN

class AE(Map):
    '''Autoencoder.
    '''
    def __init__(self, encoder_size, decoder_size, activation='sigmoid', initializer='default'):
        super(AE, self).__init__()
        self.encoder_size = encoder_size
        self.decoder_size = decoder_size
        self.activation = activation
        self.initializer = initializer
        
        self.ms = self.__init_modules()
    
    def forward(self, x):
        return self.ms['decoder'](self.ms['encoder'](x))
    
    def encode(self, x, returnnp=False):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=self.dtype, device=self.device)
        return self.ms['encoder'](x).cpu().detach().numpy() if returnnp else self.ms['encoder'](x)
    
    def decode(self, x, returnnp=False):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=self.dtype, device=self.device)
        return self.ms['decoder'](x).cpu().detach().numpy() if returnnp else self.ms['decoder'](x)
    
    def __init_modules(self):
        modules = torch.nn.ModuleDict()
        modules['encoder'] = FNN(self.encoder_size, self.activation, self.initializer)
        modules['decoder'] = FNN(self.decoder_size, self.activation, self.initializer)         
        return modules