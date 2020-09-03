from collections import namedtuple
from scipy.special import expit
import numpy as np
from .mapping import Mapping

class Activation(Mapping):
    pass

# Activation = namedtuple("Activation", ["forward", "backward"])

class Relu(Activation):
    @staticmethod
    def forward(x):
        return np.where(x>0, x, 0)

    @staticmethod
    def backward(x):
        return np.where(x.T>0, 1, 0)[..., None]*np.eye(x.shape[0])[None, ...]

# Jacobian == samples X out X in 
class Softmax:#(Activation):

    @staticmethod
    def forward(x):
        e = np.exp(x)
        return e/np.sum(e, axis=0, keepdims=True)

    @classmethod
    def backward(cls, x):
        s = cls.forward(x)
        return s.T[..., None]*(np.identity(x.shape[0])[None, ...]-s.T[:,None,:])

#np.mul.outer(_softmax(x), np.identity(x.size)-_softmax(x)))
