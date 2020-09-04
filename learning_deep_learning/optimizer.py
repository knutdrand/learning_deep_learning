from dataclasses import dataclass

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
