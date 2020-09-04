import numpy as np

from learning_deep_learning.model import CompositeAlinearModel, DoubleModel, LinearCompModel
from learning_deep_learning.optimizer import Optimizer
from learning_deep_learning.mapping import AffineMapping
from learning_deep_learning.compositemapping import LinearCompositeMapping
from .util import *

@pytest.fixture
def true_affine():
    return AffineMapping(np.array([[1, 2, 3], [4, 5, 6]], dtype="float"), np.array([[0.1], [-0.05]]))

@pytest.fixture
def affine():
    return AffineMapping(np.array([[10, 5, 2], [-3, 13, 1]], dtype="float"), 10*np.ones((2,1)))

@pytest.fixture
def true_affine2():
    return AffineMapping(np.array([[1, 2]], dtype="float"), np.array([[0.1]]))

@pytest.fixture
def affine2():
    return AffineMapping(np.array([[10, 5]], dtype="float"), 10*np.ones((1,1)))


def test_train_train(affine, true_affine):
    true_model = CompositeAlinearModel(true_affine) 
    X, y = true_model.generate_data(1000, 0)
    assert true_model.get_mean_loss(X, y) == 0
    model = CompositeAlinearModel(affine)
    Optimizer(model).train(X, y, 10000, 1)
    assert np.allclose(model.predict(X), y)
    # assert np.allclose(model.W, true_W2)
    # assert np.allclose(model.B, np.array([[0.05], [-0.05]]))

def test_gradient_double(affine, affine2):
    model = DoubleModel(affine, affine2)
    X, y = model.generate_data(1, 0)
    true_y = y-0.5
    gradient1, gradient2 = model.get_gradient(X, true_y)
    assert np.array_equal(gradient1["W"], (affine2.W*X).T)
    true_grad2 = affine.forward(X).T
    assert np.array_equal(gradient2["W"], true_grad2)

def test_train_double(affine, true_affine, affine2, true_affine2):
    true_model = DoubleModel(true_affine, true_affine2) 
    X, y = true_model.generate_data(1000, 0)
    assert true_model.get_mean_loss(X, y) == 0
    model = DoubleModel(affine, affine2)
    Optimizer(model).train(X, y, 10000, 0.003)
    assert np.allclose(model.predict(X), y)

def test_train_linear(affine, true_affine, affine2, true_affine2):
    true_model = LinearCompModel(LinearCompositeMapping([true_affine, true_affine2]))
    true_model = LinearCompModel(LinearCompositeMapping([affine, affine2]))
    X, y = true_model.generate_data(1000, 0)
    assert true_model.get_mean_loss(X, y) == 0
    model = DoubleModel(affine, affine2)
    Optimizer(model).train(X, y, 10000, 0.003)
    assert np.allclose(model.predict(X), y)

    # assert np.allclose(model.W, true_W2)
    # assert np.allclose(model.B, np.array([[0.05], [-0.05]]))
