from dataclasses import dataclass
import numpy as np

from .mapping import AffineMapping
from .activation import Softmax, Activation
from .loss import Loss, MSE

class Model:
    def get_mean_loss(self, X, y):
        return np.mean(self.loss(y).forward(self.predict(X)))

    def generate_data(self, n, epsilon=1):
        np.random.seed(42)
        X = np.random.rand(self.input_dim(), n)-0.5
        y = self.predict(X)+(np.random.rand(1, n)-0.5)*epsilon
        return X, y

    def update_model(self, gradients, rate=0.01):
        for k, v in gradients.items():
            tmp = getattr(self, k)
            assert tmp.shape==gradients[k].shape, (k, tmp, gradient[k])
            tmp -= gradient[k]*rate

@dataclass
class CompositeAlinearModel(Model):
    affine_mapping: AffineMapping
    activation: Activation=Softmax
    loss: Loss=MSE

    def predict(self, x):
        z = self.affine_mapping.forward(x)
        return self.activation.forward(z)
    
    def get_gradient(self, x, y):
        z = self.affine_mapping.forward(x)
        yhat = self.activation.forward(z)
        d_loss_on_e = self.loss(y).backward(yhat)
        d_e_on_z = self.activation.backward(z)

        J = d_loss_on_e @ d_e_on_z
        return self.affine_mapping.get_gradient(x, J)

    def update_model(self, gradients, rate=0.01):
        self.affine_mapping.update(gradients, rate=0.01)

    def input_dim(self):
        return self.affine_mapping.W.shape[1]
