import numpy as np

from learning_deep_learning.affinemodel import SimpleAffineModel, AffineModel
from .util import *

def test_train_simple(W, true_W, B, true_B):
    X, y = SimpleAffineModel(true_W, true_B).generate_data(10000, 0.001)
    model = SimpleAffineModel(W, B)
    model.train(X, y, 1000)
    assert np.allclose(model.W, true_W)
    assert np.allclose(model.B, true_B)

def test_train(W2, true_W2, B2, true_B2):
    X, y = AffineModel(true_W2, true_B2).generate_data(1000, 0.0001)
    model = AffineModel(W2, B2)
    model.train(X, y, 1000, 1)
    assert np.allclose(model.W, true_W2)
    assert np.allclose(model.B, true_B2)
