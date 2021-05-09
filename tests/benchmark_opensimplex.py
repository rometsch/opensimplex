
import sys

from opensimplex import OpenSimplex
import numpy as np

class Benchmark:
    def __init__(self):
        self.simplex = OpenSimplex(seed=0)
        # trigger compilation
        x = np.linspace(0, 1, 10)
        self.simplex.noise2d(x, x)
        self.simplex.noise3d(x, x, x)
        self.simplex.noise4d(x, x, x, x)

    def run(self, number=1000000):
        x = np.linspace(0, 1, number)
        self.simplex.noise2d(x, x)
        self.simplex.noise3d(x, x, x)
        self.simplex.noise4d(x, x, x, x)

if __name__ == "__main__":
    import cProfile
    import pstats
    b = Benchmark()
    cProfile.run("b.run()", "tests/opensimplex.prof")
    ps = pstats.Stats("tests/opensimplex.prof").sort_stats(pstats.SortKey.CUMULATIVE)
    ps.print_stats()
