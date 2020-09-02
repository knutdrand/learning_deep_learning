from collections import namedtuple
from scipy.special import expit
import numpy as np


Activation = namedtuple("Activation", ["forward", "backward"])

relu = Activation(lambda x: np.where(x>0, x, 0),
                  lambda x: np.where(x>0, 1, 0))

logistic = Activation(expit,
                      lambda x: expit(x)*expit(-x))


_softmax = lambda x: np.exp(x)/np.sum(np.exp(x), axis),
softmax = Activation(_softmax,
                     lambda x: np.mul.outer(_softmax(x), np.identity(x.size)-_softmax(x)))
