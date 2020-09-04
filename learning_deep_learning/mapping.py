from dataclasses import dataclass
import numpy as np

@dataclass
class Mapping:
    def forward(X):
        """
        input: inXsamples
        output: outXsamples
        """
        raise NotImplemented

    def backward(X):
        """
        input: inXsamples
        output: samples/1 X out X in
        """
        raise NotImplemented

@dataclass
class AffineMapping:
    W: np.array
    B: np.array

    def forward(self, x):
        return self.W @ x + self.B

    def backward(self, X):
        return self.W[None, ...]  # should maybe be copied

    def get_gradient(self, x, J):
        return {"W": np.mean(J*x.T[..., None], axis=0).T,
                "B": np.mean(J, axis=0).T}

    def update(self, gradients, rate=0.01):
        self.W -= gradients["W"]*rate
        self.B -= gradients["B"]*rate
