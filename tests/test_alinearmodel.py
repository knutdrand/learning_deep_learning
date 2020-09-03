import numpy as np

from learning_deep_learning.alinearmodel import SimpleAlinearModel, AlinearModel
from .util import *

def test_simple_train(W, true_W, B, true_B):
    X, y = SimpleAlinearModel(true_W, np.array([[0.05]])).generate_data(1000, 0)
    model = SimpleAlinearModel(W, np.array([[0.5]]))
    model.train(X, y, 1000, 1)
    assert np.allclose(model.W, true_W)
    assert np.allclose(model.B, 0.05)


def test_train_train(W2, true_W2):
    true_model = AlinearModel(true_W2, np.array([[0.1], [-0.05]]))
    X, y = true_model.generate_data(1000, 0)
    assert true_model.get_mean_loss(X, y) == 0
    model = AlinearModel(W2, 10*np.ones((2,1)))
    model.train(X, y, 10000, 1)
    assert np.allclose(model.predict(X), y)
    # assert np.allclose(model.W, true_W2)
    # assert np.allclose(model.B, np.array([[0.05], [-0.05]]))
