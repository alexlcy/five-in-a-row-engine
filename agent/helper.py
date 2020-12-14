from math import sqrt
from numba import jit
import numpy as np
from numpy import sqrt

@jit(nopython=True)
def unvisited_nodes(mat):
    max_distance = 2
    zeros = np.where(mat == 0)
    vacant = list(zip(zeros[0],zeros[1]))
    non_zeros = np.where(mat != 0)
    occupied =list(zip(non_zeros[0],non_zeros[1]))

    subset = []
    for pos in vacant:
        for loc in occupied:
            if distance(pos, loc) <= sqrt(2 * max_distance ** 2):
                subset.append(pos)
                break
    return subset

@jit(nopython=True)
def distance(pos1, pos2):
    return sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)