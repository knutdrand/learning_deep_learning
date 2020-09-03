import pytest
import numpy as np
from learning_deep_learning.activation import softmax

def test_softmax_forward():
    t=np.log([[25], [75]])
    assert np.allclose(softmax.forward(t), [[0.25], [0.75]])

def test_softmax_backward():
    t = np.log([[25], [75]])
    r = np.array([[0.25], [0.75]])
    J = softmax.backward(t)
    v = 0.25*0.75
    true_J = [[0.25*0.75, -v],
              [-v, 0.25*0.75]]
    assert np.allclose(J, true_J)


