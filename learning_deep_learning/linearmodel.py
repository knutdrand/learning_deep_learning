from dataclasses import dataclass, asdict
import numpy as np


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

def mse(a, b):
    return np.mean((a-b)**2)


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
        loss = mse(predicted, y)
        d_loss_on_W = 2*np.mean((predicted-y)*x, axis=1)
        return d_loss_on_W[None, :]

    def update_model(self, gradient, rate=0.01):
        self.W -= gradient*rate

    def train(self, X, y, n=1):
        for _ in range(n):
            gradient = self.get_gradient(X, y)
            self.update_model(gradient, 0.5)
            print(asdict(self))

    def generate_data(self, n, epsilon=1):
        X = np.random.rand(self.W.shape[1], n)
        y = self.predict(X)+(np.random.rand(1, n)-0.5)*epsilon
        return X, y
