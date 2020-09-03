import numpy as np

from learning_deep_learning.model import CompositeAlinearModel
from learning_deep_learning.optimizer import Optimizer
from learning_deep_learning.mapping import AffineMapping
from .util import *

@pytest.fixture
def true_affine():
    return AffineMapping(np.array([[1, 2, 3], [4, 5, 6]], dtype="float"), np.array([[0.1], [-0.05]]))

@pytest.fixture
def affine():
    return AffineMapping(np.array([[10, 5, 2], [-3, 13, 1]], dtype="float"), 10*np.ones((2,1)))

def test_train_train(affine, true_affine):
    true_model = CompositeAlinearModel(true_affine) 
    X, y = true_model.generate_data(1000, 0)
    assert true_model.get_mean_loss(X, y) == 0
    model = CompositeAlinearModel(affine)
    Optimizer(model).train(X, y, 10000, 1)
    assert np.allclose(model.predict(X), y)
    # assert np.allclose(model.W, true_W2)
    # assert np.allclose(model.B, np.array([[0.05], [-0.05]]))
