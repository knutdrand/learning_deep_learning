import dataclasses import dataclass

@dataclass
class Tensor:
    rank: tuple
    data: np.array

    
