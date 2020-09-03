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
        # Error = samples
        z = super().predict(x)
        f = 2*error.T[..., None] @ self.activation.backward(z)
        print(z.shape, error.shape, self.activation.backward(z).shape, f.shape)
        return {"W": np.mean(f*x.T[..., None], axis=0).T,
                "B": np.mean(f, axis=0)}


@dataclass
class AlinearModel(AffineModel):
    activation: Activation=softmax

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
        f = self.activation.backward(super().predict(x))
        return {"W": np.mean(f*x, axis=1)[None, :],
                "B": np.mean(f)}
