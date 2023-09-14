"""
@author: jpzxshi
"""
from .module import Module
from .module import Map
from .module import Algorithm
from .fnn import FNN
from .hnn import HNN
from .sympnet import LASympNet
from .sympnet import GSympNet
from .sympnet import ESympNet
from .seq2seq import S2S
from .deeponet import DeepONet
from .autoencoder import AE
from .inn import INN
from .pnn import PNN
from .pnn import AEPNN
from .mionet import MIONet

__all__ = [
    'Module',
    'Map',
    'Algorithm',
    'FNN',
    'HNN',
    'LASympNet',
    'GSympNet',
    'ESympNet',
    'S2S',
    'DeepONet',
    'AE',
    'INN',
    'PNN',
    'AEPNN',
    'MIONet',
]