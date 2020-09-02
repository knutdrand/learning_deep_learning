import pytest
import numpy as np
from learning_deep_learning.linearmodel import SimpleLinearModel
from .util import *

def test_predict_simple_linear(W):
    x = np.array([[10, 20, 30]]).T
    res = SimpleLinearModel(W).predict(x)
    wanted = np.array([[40+100+180]]).T
    assert np.array_equal(res, wanted)

def test_multi_gradient(W):
    x = np.array([[10, 20, 30],
                  [3, 2, 1]]).T
    
    y = np.array([310, 20])
    delta = SimpleLinearModel(W.copy()).get_gradient(x, y)
    print(np.array([[10, 8]])*x)
    true = 2*(np.sum(np.array([[10, 8]])*x, axis=1)/2)[None, :]
    assert np.array_equal(delta, true)

def test_one_gradient(W):
    x = np.array([[10, 20, 30]]).T
    y = 310
    delta = SimpleLinearModel(W.copy()).get_gradient(x, y)
    assert np.array_equal(delta, 2*x.T*10)


def test_update(W):
    model = SimpleLinearModel(W.copy())
    model.update_model(np.array([[100., 200., 300.]]))
    assert np.array_equal(model.W, W-np.array([[1, 2, 3]]))

def test_train(W, true_W):
    X, y = SimpleLinearModel(true_W).generate_data(10000, 0.001)
    model = SimpleLinearModel(W)
    model.train(X, y, 1000)
    assert np.allclose(model.W, true_W)

def _test_predict_linear():
    W = np.array([[1, 2, 3], [4, 5, 6]])
    x = np.array([[10, 20, 30]]).T
    res = LinearModel(W).predict(x)
    wanted = np.array([[10+40+90, 
                        40+100+180]]).T
    assert np.array_equal(res, wanted)
        
