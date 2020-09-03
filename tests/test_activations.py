import pytest
import numpy as np
from learning_deep_learning.activation import Softmax

def test_softmax_forward():
    t=np.log([[25], [75]])
    assert np.allclose(Softmax.forward(t), [[0.25], [0.75]])

def test_softmax_backward():
    t = np.log([[25], [75]])
    r = np.array([[0.25], [0.75]])
    J = Softmax.backward(t)
    v = 0.25*0.75
    true_J = [[0.25*0.75, -v],
              [-v, 0.25*0.75]]
    assert np.allclose(J, true_J)


