
import sys

from opensimplex import OpenSimplex
import numpy as np

if sys.version_info[0] < 3:
    _range = xrange
else:
    _range = range


from subprocess import run

with open("tests/old_opensimplex.py", "w") as outfile:
    res = run(["git", "show", "81360ca728a26cf4ea2bb3287d500ef9bd4c80a9:opensimplex/opensimplex.py"], stdout=outfile)
from old_opensimplex import OpenSimplex as OldOpenSimplex

class Benchmark:
    def __init__(self):
        self.oldsimplex = OldOpenSimplex(seed=0)

    def run(self, number=1000000):
        for i in _range(number):
            self.oldsimplex.noise2d(0.1, 0.1)
            self.oldsimplex.noise3d(0.1, 0.1, 0.1)
            self.oldsimplex.noise4d(0.1, 0.1, 0.1, 0.1)

if __name__ == "__main__":
    import cProfile
    import pstats
    b = Benchmark()
    cProfile.run("b.run()", "tests/old_opensimplex.prof")
    ps = pstats.Stats("tests/old_opensimplex.prof").sort_stats(pstats.SortKey.CUMULATIVE)
    ps.print_stats()
