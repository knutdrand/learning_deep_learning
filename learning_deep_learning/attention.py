from dataclasses import dataclass
import numpy as np
from .activation import Softmax, SeqSoftmax
from .mapping import Mapping

def T(array):
    return array.transpose([0,2,1])

@dataclass
class Attention(Mapping):
    W_K: np.array
    W_Q: np.array
    W_V: np.array

    def input_dim(self):
        return self.W_K.shape[1]

    def forward(self, X):
        """
        softmax(Q(x)K(x)')*V(x)'
        """
        keys = self.W_K @ X
        queries = self.W_Q @ X
        values = self.W_V @ X
        scores = Softmax.forward(T(keys) @ queries)
        # print(scores)
        print(scores.shape, keys.shape)
        return values @ scores

    def get_gradient(self, X, J):
        K = self.W_K @ X
        Q = self.W_Q @ X
        V = self.W_V @ X
        S = T(K) @ Q
        A = Softmax.forward(S)
        
        d_K = T(Q) @ X
        d_Q = T(X) @ K
        d_V = A
        d_A = V
        d_S = Softmax.backward(S)
        return {"V": J @ V}


    def backward(self, y):
        """
        S = (Kx).T@Qx
        d softmax(S)@Vx/dx
        = d softmax(S)/dx@Vx + softmax(S)@dVx/dx
        = d softmax(S)/dx@Vx + softmax(S)@dV
        = softmax'(S) dS/dx@Vx + softmax(S)@dV
        """
        pass
        

class Weights(Innerprod):
    def forward(self, X):
        return SeqSoftmax.forward(super().forward(X))

    def get_gradient(self, X, J):
        S = super().forward(X)
        nJ = Softmax.backward(S)
        dWq = self.d_W_q(X, J)
        

class Innerprod(Attention):

    def key_dim(self):
        return self.W_K.shape[0]

    def forward(self, X):
        keys = self.W_K @ X
        queries = self.W_Q @ X
        return T(keys) @ queries

    def dW_q(self, X, J):
        n, input_dim, L = X.shape
        keys = self.W_K @ X
        assert keys.shape == (n, self.key_dim(), L)
        # d_S_wq = (X[..., None, None]*keys[:, None, None, ...]).swapaxes(1, 3).swapaxes(-2, -1)
        d_S_wq = np.einsum("...ai,...bj->...ijab", keys, X)
        assert d_S_wq.shape == (n, L, L, self.key_dim(), input_dim), (d_S_wq.shape, (n, L, L, self.key_dim(), input_dim))
        # dS_tijkl,sample t influence of X_kl on S_ij
        # J_tab,sample t  influence of S_ab on L
        # dL_tcd=sample t influence of X_cd on L = sum_ij(dS_tijcd*J_tij)

        return np.einsum("...ijcd,...ij", d_S_wq, J).mean(axis=0)

    def dW_k(self, X, J):
        n, input_dim, L = X.shape
        queries = self.W_K @ X
        assert queries.shape == (n, self.key_dim(), L)
        d_S_wk = np.einsum("...aj,...bi->...ijab", queries, X)
        assert d_S_wk.shape == (n, L, L, self.key_dim(), input_dim), (d_S_wk.shape, (n, L, L, self.key_dim(), input_dim))
        return np.einsum("...ijcd,...ij", d_S_wk, J).mean(axis=0)

    def get_gradient(self, X, J):
        n, input_dim, L = X.shape
        assert J.shape==(n, L, L), (J.shape,(n, L, L))

        S = self.forward(X)
        assert S.shape==(n, L, L), (S.shape, (n, L, L))
        return {"W_Q": self.dW_q(X, J),
                "W_K": self.dW_k(X, J)}

    def input_dim(self):
        return self.W_K.shape[1]
        
    def update(self, gradients, rate=0.01):
        updates = {}
        for key, gradient in gradients.items():
            d = -gradients[key]*rate
            tmp = getattr(self, key)
            tmp += d
            updates[key]=d
        return updates



@dataclass
class Scores(Mapping):
    S: np.array

    def input_dim(self):
        return 3

    def forward(self, X):
        return np.array(X.shape[0]*[self.S])
    
    def get_gradient(self, X, J):
        return {"S": np.mean(J, axis=0)}

    def update(self, gradients, rate=0.01):
        d = -gradients["S"]*rate
        self.S += d
        return {"S": d}
