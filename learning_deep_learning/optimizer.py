from dataclasses import dataclass
import numpy as np

from .model import Model

@dataclass
class Optimizer:
    model: Model

    def train(self, X, y, n=1, rate=0.5):
        for counter in range(n):
            gradient = self.model.get_gradient(X, y)
            if counter % 100 == 0:
                print(gradient)
                print(self.model)
                print(self.model.get_mean_loss(X, y))
            self.model.update_model(gradient, rate)

class BBOptimizer(Optimizer):
    #Barzilai–Borwein

    def update_gradients(self,prev_gradients, gradients, d):
        updated = {}
        for key in gradients:
            u = np.sum(d[key]*(gradients[key]-prev_gradients[key]))
            l = np.sum((gradients[key]-prev_gradients[key])**2)
            f = u/l if (l > 10e-25) else 1
            updated[key] = f*gradients[key]
        return updated
    
    def train(self, X, y, n=1, rate=0.5):
        gradient = self.model.get_gradient(X, y)
        d = self.model.update_model(gradient, rate)
        for counter in range(n):
            prev_gradient = gradient
            gradient = self.model.get_gradient(X, y)
            updated = self.update_gradients(prev_gradient, gradient, d)
            if counter % 10 == 0:
                print(gradient)
                print(self.model)
                print(self.model.get_mean_loss(X, y))
            d = self.model.update_model(updated, 1)
