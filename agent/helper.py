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


def find_children_priority(state, subset, player):
    ''' this is to select which children to expand, different scores are added to each move
        if the move leads to different scenarios:
        1. 5 (inf)
        2. break 5 (20000)
        3. open 4 (10000)
        4. close 4 and open 3 (5000)
        5. break open 4 (4000)
        6. break (close 4 and open 3) (3000)
        7. open 3 x 2 (2000)
        8. break (open 3 x 2) (1000)
        9. open 2 x 3 (300)
        10. open 2 x 2 (200)
        11. open 2 (100)
        '''
    n = state.shape[0]
    scores = {}
    offset = {'h': (0, 1), 'v': (1, 0), 'd': (1, 1), 'ad': (-1, 1)}
    for pos in subset:
        i = pos[0]
        j = pos[1]
        scores[(i, j)] = 0
        closed_4 = False
        break_closed_4 = False
        for os in offset.values():
            # Linking Count
            i1 = i + os[0]
            j1 = j + os[1]
            i2 = i - os[0]
            j2 = j - os[1]
            linking = 1
            is_open = True
            while i1 >= 0 and j1 >= 0 and i1 < n and j1 < n:
                if state[i1][j1] == player:
                    linking += 1
                    i1 += os[0]
                    j1 += os[1]
                elif state[i1][j1] != 0:
                    is_open = False
                    break
                else:
                    break
            while i2 >= 0 and j2 >= 0 and i2 < n and j2 < n:
                if state[i2][j2] == player:
                    linking += 1
                    i2 -= os[0]
                    j2 -= os[1]
                elif state[i2][j2] != 0:
                    is_open = False
                    break
                else:
                    break
            if i1 < 0 or j1 < 0 or i1 == n or j1 == n or i2 < 0 or j2 < 0 or i2 == n or j2 == n:
                is_open = False
            if linking >= 5:
                scores[(i, j)] = float('inf')
                break
            elif linking == 4 and is_open:
                scores[(i, j)] += 10000
            elif linking == 4:
                closed_4 = True
            elif linking == 3 and closed_4:
                scores[(i, j)] += 5000
            elif linking == 3 and is_open:
                scores[(i, j)] += 1000
            elif linking == 2 and is_open:
                scores[(i, j)] += 100
            # Breaking Count
            i1 = i + os[0]
            j1 = j + os[1]
            i2 = i - os[0]
            j2 = j - os[1]
            breaking = 1
            is_open = True
            while i1 >= 0 and j1 >= 0 and i1 < n and j1 < n:
                if state[i1][j1] == -player:
                    breaking += 1
                    i1 += os[0]
                    j1 += os[1]
                elif state[i1][j1] != 0:
                    is_open = False
                    break
                else:
                    break
            while i2 >= 0 and j2 >= 0 and i2 < n and j2 < n:
                if state[i2][j2] == -player:
                    breaking += 1
                    i2 -= os[0]
                    j2 -= os[1]
                elif state[i2][j2] != 0:
                    is_open = False
                    break
                else:
                    break
            if i1 < 0 or j1 < 0 or i1 == n or j1 == n or i2 < 0 or j2 < 0 or i2 == n or j2 == n:
                is_open = False
            if breaking >= 5:
                scores[(i, j)] = 20000
                break
            elif breaking == 4 and is_open:
                scores[(i, j)] += 4000
            elif breaking == 4:
                break_closed_4 = True
            elif breaking == 3 and break_closed_4:
                scores[(i, j)] += 3000
            elif breaking == 3 and is_open:
                scores[(i, j)] += 500
    sorted_score = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [i[0] for i in sorted_score[:5]]
