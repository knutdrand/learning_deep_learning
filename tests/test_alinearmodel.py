import numpy as np

from learning_deep_learning.alinearmodel import SimpleAlinearModel
from .util import *

def test_train(W, true_W, B, true_B):
    X, y = SimpleAlinearModel(true_W, np.array([[0.05]])).generate_data(1000, 0)
    model = SimpleAlinearModel(W, np.array([[0.5]]))
    model.train(X, y, 1000, 1)
    assert np.allclose(model.W, true_W)
    assert np.allclose(model.B, 0.05)
