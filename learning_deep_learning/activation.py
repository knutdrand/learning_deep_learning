from collections import namedtuple
from scipy.special import expit
import numpy as np


Activation = namedtuple("Activation", ["forward", "backward"])

relu = Activation(lambda x: np.where(x>0, x, 0),
                  lambda x: np.where(x>0, 1, 0))

logistic = Activation(expit,
                      lambda x: expit(x)*expit(-x))
