from Filters import Filter
from typing import Union, Any, Callable
from random import Random

class Composite:
    """
    Composes filters into one unit and randomly applies a filter when the transform() method is called
    """
    def __init__(self, filters:list[Filter], useBBox=False, seed:Union[None, Any]=None, probablityFunction:Callable[[float], float] = lambda x: x) -> None:
        self.filters = filters
        self.useBBox = useBBox

        if seed:
            for f in filters:
                f.setSeed(seed)
            self.rand = Random(seed)
        else:
            self.rand = Random()

        self.probablityFunction = probablityFunction

    def pickIndex(self):
        x = self.rand.random()
        y = self.probablityFunction(x) * len(self.filters)

        return min(list(range(y)), key=lambda n: abs(n-x))