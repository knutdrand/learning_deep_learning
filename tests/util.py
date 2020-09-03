import pytest
import numpy as np

@pytest.fixture
def W():
    return np.array([[4., 5., 6.]])

@pytest.fixture
def true_W():
    return np.array([[10., 7., 19.]])

@pytest.fixture
def B():
    return np.array([[100.]])

@pytest.fixture
def true_B():
    return np.array([[50.]])



@pytest.fixture
def W2():
    return np.array([[10, 5, 2], [-3, 13, 1]], dtype="float")

@pytest.fixture
def true_W2():
    return np.array([[1, 2, 3], [4, 5, 6]], dtype="float")

@pytest.fixture
def B2():
    return np.array([[100.], [20.]])

@pytest.fixture
def true_B2():
    return np.array([[50.], [60.]])


