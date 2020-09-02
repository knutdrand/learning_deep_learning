from dataclasses import dataclass

import numpy as np
from scipy.special import expit

from .affinemodel import SimpleAffineModel
from .activation import *

# dexpit = lambda x: np.exp(x)/(1+np.exp(x))**2
# relu = lambda x: np.where(x>0, x, 0)
# drelu = lambda x: np.where(x>0, 1, 0)

@dataclass
class AlinearModel(SimpleAffineModel):
    W: np.array
    B: np.array
    activation: Activation=relu
    def predict(self, x):
        p = super().predict(x)
        return self.activation.forward(p)
    
    def get_gradient(self, x, y):
        """
        z = Wx+B
        e = expit(z)-y
        l = sum[e^2]/n
        dl/dW = 2(e)*dexpit(z)*x^T
        """
        predicted = self.predict(x)
        error = predicted-y
        z = super().predict(x)
        f = 2*error*self.activation.backward(z)
        return {"W": np.mean(f*x, axis=1)[None, :],
                "B": np.mean(f)}

