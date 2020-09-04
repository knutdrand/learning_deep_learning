from dataclasses import dataclass
from operator import mul
from itertools import accumulate
from .mapping import Mapping

laccumulate=lambda *args, **kwargs: list(accumulate(*args, **kwargs))

@dataclass
class LinearCompositeMapping(Mapping):
    components: list

    def _forward(self, x):
        return laccumulate(self.components, lambda d, f: f.forward(d), initial=x)[-1]

    def forward(self, x):
        *_, res = self._forward(x)
        return res

    def backward(self, x):
        pass
    
    def get_gradient(self, x, J):
        # x1, x2, x3, x4, y
        # x2=w1(x1), x3=w2(x2), x4=w3(x3), y=w4(x4)
        # dy/x4=w4, dy/x3=dy/x4*x4/x3
        # dy/xi = dy/dx_(i+1)dx_(i+1)/dx = dy/dx_(i+1)*w_i

        forward_states = list(self._forward(x))
        differentials = (component.backward(y) for component, y in
                         reversed(zip(self.components, forward_states)))
        Js = laccumulate(differentials, np.matmul)
        gradients = []
        for J, component, state in zip(Js, reversed(self.components), reversed(forward_states)):
            gradients.append(component.get_gradient(state, J))
        return gradients[::-1]
    
    def update(self, gradients, rate=0.01):
        for comp, gradient in zip(self.components, gradients):
            comp.update(gradient)
            
    def input_dim(self):
        return self.components[0].W.shape[1]
