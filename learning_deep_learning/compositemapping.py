from dataclasses import dataclass
from itertools import accumulate
from .mapping import Mapping

@dataclass
class LinearCompositeMapping(Mapping):
    components: list

    def _forward(self, x):
        return accumulate(self.components, lambda d, f: f(d), initial=x))[-1]

    def forward(self, x):
        *self.state, res = self._forward
        return res

    def backward(self, x, y):
        for s, mapping in zip(reversed(self.state), self.components):
            
        accumulate(reversed(self.state,  )
        results = self.forward(

