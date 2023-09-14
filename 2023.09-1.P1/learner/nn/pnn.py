"""
@author: jpzxshi
"""
import torch
from .module import Map, Algorithm

class PNN(Map):
    '''INN-based Poisson neural network.
    '''
    def __init__(self, inn, sympnet, recurrent=1):
        super(PNN, self).__init__()
        self.inn = inn
        self.sympnet = sympnet
        self.recurrent = recurrent
        
        self.dim = sympnet.dim
    
    def forward(self, x):
        x = self.inn(x)
        for i in range(self.recurrent):
            x = self.sympnet(x)
        return self.inn.inverse(x)
    
    def predict(self, x, steps=1, keepinitx=False, returnnp=False):
        x = self._to_tensor(x)
        size = len(x.size())
        pred = [self.inn(x)]
        for _ in range(steps):
            pred.append(self.sympnet(pred[-1]))
        pred = list(map(self.inn.inverse, pred))
        if keepinitx:
            steps = steps + 1
        else:
            pred = pred[1:]
        res = torch.cat(pred, dim=-1)
        if steps > 1:
            res = res.view([-1, steps, self.dim][2 - size:])
        return res.cpu().detach().numpy() if returnnp else res
    
class AEPNN(Algorithm):
    '''Autoencoder-based Poisson neural network.
    '''
    def __init__(self, ae, sympnet, lam=1, recurrent=1):
        super(AEPNN, self).__init__()
        self.ae = ae
        self.sympnet = sympnet
        self.lam = lam
        self.recurrent = recurrent
        
        self.dim = ae.encoder_size[0]
    
    def criterion(self, X, y):
        X_latent, y_latent = self.ae.encode(X), self.ae.encode(y)
        X_latent_step = X_latent
        for i in range(self.recurrent):
            X_latent_step = self.sympnet(X_latent_step)
        symp_loss = torch.nn.MSELoss()(X_latent_step, y_latent)
        ae_loss = torch.nn.MSELoss()(self.ae.decode(X_latent), X) + torch.nn.MSELoss()(self.ae.decode(y_latent), y)
        return symp_loss + self.lam * ae_loss
    
    def predict(self, x, steps=1, keepinitx=False, returnnp=False):
        x = self._to_tensor(x)
        size = len(x.size())
        pred = [self.ae.encode(x)]
        for _ in range(steps):
            pred.append(self.sympnet(pred[-1]))
        pred = list(map(self.ae.decode, pred))
        if keepinitx:
            steps = steps + 1
        else:
            pred = pred[1:]
        res = torch.cat(pred, dim=-1)
        if steps > 1:
            res = res.view([-1, steps, self.dim][2 - size:])
        return res.cpu().detach().numpy() if returnnp else res