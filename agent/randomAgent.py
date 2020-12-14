from agent.base import Agent

import random
from agent.helper import unvisited_nodes

class RandomAgent(Agent):

    def select_move(self, mat):
        subset = unvisited_nodes(mat)

        choice = random.choice(subset)
        mat[choice[0]][choice[1]] = -1
        return mat


