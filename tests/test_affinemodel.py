import numpy as np

from learning_deep_learning.affinemodel import AffineModel
from .util import *

def test_train(W, true_W, B, true_B):
    X, y = AffineModel(true_W, true_B).generate_data(10000, 0.001)
    model = AffineModel(W, B)
    model.train(X, y, 1000)
    assert np.allclose(model.W, true_W)
    assert np.allclose(model.B, true_B)
