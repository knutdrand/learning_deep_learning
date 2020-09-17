import numpy as np

from learning_deep_learning.model import SeqModel
from learning_deep_learning.optimizer import Optimizer, BBOptimizer
from learning_deep_learning.attention import Attention, Innerprod, Scores
from learning_deep_learning.activation import Relu, Softmax
from .util import *

@pytest.fixture
def attention():
    W_K = np.array([[2., 3, 2], [1, 2, 3]])
    W_Q = np.array([[4., 3, 2], [1, 2, 2]])
    W_V = np.arange(6, dtype="float").reshape((2,3))
    return Attention(W_K, W_Q, W_V)

@pytest.fixture
def innerprod():
    W_K = np.array([[2., 3, 2], [1, 2, 3]])
    W_Q = np.array([[4., 3, 2], [1, 2, 2]])
    W_V = np.arange(6, dtype="float").reshape((2,3))
    return Innerprod(W_K, W_Q, W_V)


def test_generate_shape(attention):
    n, L = (10, 4)
    model = SeqModel(attention)
    X, y = model.generate_data(n, L)
    assert X.shape == (10, 3, 4)
    # assert y.shape == (10, 2, 4)


def test_update_W_q_inner(innerprod):
    n, L = (1000, 4)
    true_innerprod = Innerprod(innerprod.W_K, innerprod.W_Q.copy(), innerprod.W_V)
    true_innerprod.W_Q += 0.01*np.array([[1, -1, 10], [-10, 10, -1]])
    model = SeqModel(innerprod)
    true_model = SeqModel(true_innerprod)
    X, y = true_model.generate_data(n, L, 0)
    assert X.shape == (n, 3, 4)
    assert y.shape == (n, 4, 4)
    BBOptimizer(model).train(X, y, 200, 0.005)
    print(true_model)
    assert np.allclose(model.predict(X), y)

def test_scores():
    n, L = (1000, 4)
    true_model = SeqModel(Scores(np.arange(L*L, dtype="float").reshape((L, L))))
    model = SeqModel(Scores(np.arange(L*L, dtype="float").reshape((L, L)).T))
    X, y = true_model.generate_data(n, L, 0)
    assert y.shape == (n, L, L)
    BBOptimizer(model).train(X, y, 3, 0.1)
    print(true_model)
    assert np.allclose(model.predict(X), y)
