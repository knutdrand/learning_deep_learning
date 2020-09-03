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


class AffineMapping:
    W: np.array
    B: np.array

    def forward(X):
        return self.W.dot(x)+self.B

    def backward(X):
        return W.T[None, ...]  # should maybe be copied
