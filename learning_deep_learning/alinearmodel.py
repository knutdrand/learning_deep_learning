from dataclasses import dataclass

import numpy as np
from scipy.special import expit

from .affinemodel import SimpleAffineModel, AffineModel
from .activation import *

# dexpit = lambda x: np.exp(x)/(1+np.exp(x))**2
# relu = lambda x: np.where(x>0, x, 0)
# drelu = lambda x: np.where(x>0, 1, 0)

@dataclass
class SimpleAlinearModel(SimpleAffineModel):
    activation: Activation=Relu
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
        # Error = samples
        J = self.activation.backward(super().predict(x))
        f = 2*error.T[..., None] @ J
        return {"W": np.mean(f*x.T[..., None], axis=0).T,
                "B": np.mean(f, axis=0)}


@dataclass
class AlinearModel(AffineModel):
    activation: Activation=Softmax

    def predict(self, x):
        p = super().predict(x)
        return self.activation.forward(p)
    
    def get_gradient(self, x, y):
        """
        mse(softmax(Wx+B), y)
        dL/dW = dL/de*de/dz*dz/dW
              = 1Xm * mXm * mXWs
        z = Wx+B
        e = expit(z)-y
        l = sum[e^2]/n
        dl/dW = 2(e)*dexpit(z)*x^T
        """
        predicted = self.predict(x)
        d_loss_on_e = self.loss.backward(predicted, y)
        d_e_on_z = self.activation.backward(super().predict(x))
        f = d_loss_on_e @ d_e_on_z
        # f = samples, out, in
        # x = n,samples
        return {"W": np.mean(f*x.T[..., None], axis=0).T,
                "B": np.mean(f, axis=0).T}
