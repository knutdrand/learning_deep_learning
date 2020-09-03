import numpy as np
from collections import namedtuple
"""
Jacobian = out X in
Jacobian samples = out X in X samples
"""

Loss = namedtuple("Loss", ["forward", "backward"])

#Axes: outdim=1 X samples
mse = Loss(lambda p, y: np.mean((p-y)**2, axis=0, keepdims=True),
           lambda p, y: 2*(p-y).T[:, None, :]/y.shape[0])
#Axes: samples x out x in 
