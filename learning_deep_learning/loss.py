import numpy as np
from collections import namedtuple
from dataclasses import dataclass

from .mapping import Mapping

"""
Jacobian = out X in
Jacobian samples = out X in X samples
"""
@dataclass
class Loss(Mapping):
    y: np.array
    

#Loss = namedtuple("Loss", ["forward", "backward"])

#Axes: outdim=1 X samples
# mse = Loss(lambda p, y: np.mean((p-y)**2, axis=0, keepdims=True),
#           lambda p, y: 2*(p-y).T[:, None, :]/y.shape[0])

class MSE(Loss):
    def forward(self, p):
        return np.mean((p-self.y)**2, axis=0, keepdims=True)

    def backward(self, p):
        return 2*(p-self.y).T[:, None, :]/self.y.shape[0]      
    
    


#Axes: samples x out x in 
