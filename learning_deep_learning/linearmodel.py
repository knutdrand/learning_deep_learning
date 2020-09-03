from dataclasses import dataclass, asdict
import numpy as np

from .loss import Loss, mse

"""
x = [1,4
     2,5
     3,6]
y = [10, 20]
W = [100, 200, 300]
p = [a, b]

Wx = [w1.x, w2.x]


x=(3x1)
W=(2x3)
Wx=(2x1)


[ux+vy+wz-t]**2
du=2e*x
dv=2e*y
dw=2e*z
[2e*
"""
"""
d(Wx)/dW = [d([x^TW^T]^T)/dW] = x^TT
"""

@dataclass
class SimpleLinearModel:
    W: np.array

    def predict(self, x):
        assert x.shape[0] == self.W.shape[-1]
        return self.W.dot(x)

    def get_gradient(self, x, y):
        """
        l = sum[(Wx-y)^2]/n
        dl/dW = 2(Wx-y)*x^T
        """
        predicted = self.predict(x)
        d_loss_on_W = 2*np.mean((predicted-y)*x, axis=1)
        return d_loss_on_W[None, :]

    def update_model(self, gradient, rate=0.01):
        self.W -= gradient*rate

    def train(self, X, y, n=1, rate=0.5):
        for counter in range(n):
            gradient = self.get_gradient(X, y)
            self.update_model(gradient, rate)
            if counter % 1000 == 0:
                print(gradient)
                print(self)
                print

    def generate_data(self, n, epsilon=1):
        np.random.seed(42)
        X = np.random.rand(self.W.shape[1], n)-0.5
        y = self.predict(X)+(np.random.rand(1, n)-0.5)*epsilon
        return X, y



@dataclass
class LinearModel(SimpleLinearModel):
    loss = mse

    def get_gradient(self, x, y):
        """
        ((w_1x-y_1)**2+(w_2x-y_2)**2)/2
        dL/dw_1 = 2*e_1*x
        dL/dw_2 = 2*e_2*x
        """

        """
        l = loss(Wx-y)
        dl/dW = dl/de*de/dW
              = 2e*W
    
        l = sum[(Wx-y)^2]/n
        dl/dW = 2(Wx-y)*x^T
        """
        predicted = self.predict(x)
        d_loss_on_e = self.loss.backward(predicted, y)
        # X.T=samplesXin  J=samplesXoutXin
        return np.mean(x.T[:, :, None]*d_loss_on_e, axis=0).T

    def get_mean_loss(self, X, y):
        return np.mean(self.loss.forward(self.predict(X), y))

    def train(self, X, y, n=1, rate=0.5):
        for counter in range(n):
            gradient = self.get_gradient(X, y)
            self.update_model(gradient, rate)
            if counter % 500 == 0:
                print(gradient)
                print(self)
                print(self.get_mean_loss(X, y))
