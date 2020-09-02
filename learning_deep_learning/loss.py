import numpy as np
from collections import namedtuple
Loss = namedtuple("Loss", ["forward", "backward"])

mse = Loss(lambda p, y: np.mean((p-y)**2, axis=0, keepdims=True),
           lambda p, y: 2*(p-y)/y.shape[0])
