import numpy as np
from dataclasses import dataclass, asdict

from .linearmodel import SimpleLinearModel, LinearModel


@dataclass
class SimpleAffineModel(SimpleLinearModel):
    W: np.array
    B: np.array
    
    def predict(self, x):
        return self.W.dot(x)+self.B
    
    def get_gradient(self, x, y):
        """
        l = sum[(Wx+B-y)^2]/n
        dl/dW = 2(Wx+B-y)*x^T
        """
        predicted = self.predict(x)
        error = predicted-y
        d_loss_on_W = 2*np.mean(error*x, axis=1)
        d_loss_on_B = 2*np.mean(error)
        return {"W": d_loss_on_W[None, :], "B": d_loss_on_B}
    
    def update_model(self, gradient, rate=0.01):
        for k, v in gradient.items():
            tmp = getattr(self, k)
            tmp -= gradient[k]*rate


@dataclass
class AffineModel(LinearModel):
    B: np.array

    def predict(self, x):
        return self.W.dot(x)+self.B

    def get_gradient(self, x, y):
        predicted = self.predict(x)
        d_loss_on_e = self.loss.backward(predicted, y)
        dW = np.mean(x.T[:, :, None]*d_loss_on_e, axis=0).T
        # J=samplesXoutXin  B=inX1
        dB = np.mean(d_loss_on_e, axis=0, keepdims=False).T
        return {"W": dW, "B": dB}
        
    def update_model(self, gradient, rate=0.01):
        for k, v in gradient.items():
            tmp = getattr(self, k)
            assert tmp.shape==gradient[k].shape, (k, tmp, gradient[k])
            tmp -= gradient[k]*rate
