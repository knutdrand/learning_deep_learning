from dataclasses import dataclass
import numpy as np

from .mapping import AffineMapping, Mapping
from .compositemapping import LinearCompositeMapping
from .activation import Softmax, Activation
from .loss import Loss, MSE, GeneralMSE

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
class SeqModel(Model):
    mapping: Mapping
    loss: Loss=GeneralMSE

    def predict(self, x):
        return self.mapping.forward(x)

    def generate_data(self, n, L=10, epsilon=1):
        np.random.seed(42)
        X = np.random.rand(n, self.input_dim(), L)-0.5
        y = self.predict(X)
        noise = (np.random.rand(*(y.shape))-0.5)*epsilon
        return X, y + noise

    def input_dim(self):
        return self.mapping.input_dim()

    def get_gradient(self, x, y):
        yhat = self.predict(x)
        d_loss_on_e = self.loss(y).backward(yhat)
        return self.mapping.get_gradient(x, d_loss_on_e)
                
    def update_model(self, gradients, rate=0.01):
        return self.mapping.update(gradients, rate=rate)


@dataclass
class CompositeAlinearModel(Model):
    affine_mapping: AffineMapping
    activation: Activation=Softmax
    loss: Loss=MSE

    def predict(self, x):
        z = self.affine_mapping.forward(x)
        return self.activation.forward(z)
    
    def get_gradient(self, x, J):
        z = self.affine_mapping.forward(x)
        yhat = self.activation.forward(z)
        d_loss_on_e = self.loss(J).backward(yhat)
        d_e_on_z = self.activation.backward(z)
        d_loss_on_z = d_loss_on_e @ d_e_on_z
        return self.affine_mapping.get_gradient(x, d_loss_on_z)

    def update_model(self, gradients, rate=0.01):
        self.affine_mapping.update(gradients, rate=rate)

    def input_dim(self):
        return self.affine_mapping.W.shape[1]

@dataclass
class DoubleModel(Model):
    affine_mapping: AffineMapping
    affine_mapping2: AffineMapping
    loss: Loss=MSE

    def predict(self, x):
        z = self.affine_mapping.forward(x)
        return self.affine_mapping2.forward(z)
    
    def get_gradient(self, x, y):
        z = self.affine_mapping.forward(x)
        yhat = self.affine_mapping2.forward(z)
        d_loss_on_e = self.loss(y).backward(yhat)
        d_e_on_z = self.affine_mapping2.backward(z)
        J = d_loss_on_e @ d_e_on_z
        return (self.affine_mapping.get_gradient(x, J),
                self.affine_mapping2.get_gradient(z, d_loss_on_e))
                
    def update_model(self, gradients, rate=0.01):
        self.affine_mapping.update(gradients[0], rate=rate)
        self.affine_mapping2.update(gradients[1], rate=rate)

    def input_dim(self):
        return self.affine_mapping.W.shape[1]


@dataclass
class LinearCompModel(Model):
    mapping: LinearCompositeMapping
    loss: Loss=MSE

    def predict(self, x):
        return self.mapping.forward(x)
    
    def get_gradient(self, x, y):
        yhat = self.predict(x)
        d_loss_on_e = self.loss(y).backward(yhat)
        return self.mapping.get_gradient(x, d_loss_on_e)
                
    def update_model(self, gradients, rate=0.01):
        self.mapping.update(gradients, rate=rate)

    def input_dim(self):
        return self.mapping.input_dim()
