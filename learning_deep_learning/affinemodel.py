import numpy as np
from dataclasses import dataclass, asdict

from .linearmodel import SimpleLinearModel, mse


@dataclass
class AffineModel(SimpleLinearModel):
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
