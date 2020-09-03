from dataclasses import dataclass
from itertools import accumulate
from .mapping import Mapping

@dataclass
class CompositeMapping(Mapping):
    components = [] # Shoud be graph

    def forward(x):
        return list(accumulate(self.components, lambda d, f: f(d), initial=x))[-1]
        
