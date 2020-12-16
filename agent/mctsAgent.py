from agent.base import Agent
import numpy as np
import time
import random
import math
from MCTS import monte_carlo_tree_search, Node
from agent.helper import unvisited_nodes,find_children_priority
from fiveinarow import check_for_win


def pprint_tree(node, file=None, _prefix="", _last=True):
    print(_prefix, "`- " if _last else "|- ", (node.win_num, node.visit_num), sep="", file=file)
    _prefix += "   " if _last else "|  "
    child_count = len(node.children)
    for i, child in enumerate(node.children):
        _last = i == (child_count - 1)
        pprint_tree(child, file, _prefix, _last)

class MCTSNode(object):
    def __init__(self, mat, player, move=None, parent=None):
        self.game_state = mat
        self.player = player
        self.parent = parent
        self.children = []
        self.visit_num = 0
        self.win_num = 0
        self.move = move
        if self.move == None:
            self.unvisited_nodes = [(i, j) for i in range(mat.shape[0]) for j in range(mat.shape[0]) if mat[i][j] == 0]
        else:
            self.unvisited_nodes = find_children_priority(self.game_state, unvisited_nodes(mat), self.player)
        np.random.shuffle(self.unvisited_nodes)

    def expansion(self):
        choice = self.unvisited_nodes.pop()
        game_state_tmp = np.copy(self.game_state)
        game_state_tmp[choice[0]][choice[1]] = self.player * -1
        new_node = MCTSNode(game_state_tmp, self.player * -1, move=choice, parent=self)
        self.children.append(new_node)
        return new_node

    def winning_percent(self):
        return self.win_num/self.visit_num

    def can_add_child(self):
        return len(self.unvisited_nodes) > 0

    def is_terminal(self):
        result, _ = check_for_win(self.game_state, self.move)
        return result

class MCTSAgent(Agent):

    def __init__(self, simulation_number, temperature, cur_player):
        Agent.__init__(self)
        self.cur_player = cur_player
        self.simulation_number = simulation_number
        self.temperature = temperature

    def select_move(self, mat, move):
        cur_time = time.time()

        root = MCTSNode(mat, self.cur_player, move=move)
        simulation_num = 0
        for i in range(self.simulation_number):
        #while time.time() - cur_time < 10:
            node = root

            # Step 1: Selection
            while node.move is not None and (not node.can_add_child()) and check_for_win(node.game_state, node.move) is None:
                node = self.select_child(node)

            # Step 2: Expansion
            if node.can_add_child():
                node = node.expansion()

            # Step 3: Rollout and get the winner
            winner = self.rollout(node)

            # Step 4: Back propagate to update the score
            while node is not None:
                if node.player == winner:
                    node.win_num += 1
                if winner == 0:
                    node.win_num += 0.5
                node.visit_num += 1
                node = node.parent

            simulation_num +=1

        visited_num_mat = np.zeros((8,8))
        winning_num_mat = np.zeros((8,8))
        winning_percent_mat = np.zeros((8, 8))

        best_move_mat = None
        best_move = None
        best_visit_num = -1.0
        for child in root.children:
            child_visit_num = child.visit_num
            move = child.move
            visited_num_mat[move[0]][move[1]] = child.visit_num
            winning_percent_mat[move[0]][move[1]] = np.round(child.winning_percent(), 3)
            winning_num_mat[move[0]][move[1]] = child.win_num

            if child_visit_num > best_visit_num:
                best_visit_num = child_visit_num
                best_move_mat = child.game_state
                best_move = child.move

        return best_move_mat, best_move

    def select_child(self, node):
        total_rollout = sum([child.visit_num for child in node.children])
        log_rollout = np.log(total_rollout)

        best_score = -1
        best_child = None

        for child in node.children:
            win_percent = child.winning_percent()
            exploration_factor = math.sqrt(log_rollout/child.visit_num)
            uct_score = win_percent + self.temperature * exploration_factor
            if uct_score > best_score:
                best_child = child
                best_score = uct_score
        return best_child

    def rollout(self, node):
        rollout_mat = np.copy(node.game_state)
        zeros = np.where(rollout_mat == 0)
        unvisited_node_list = list(zip(zeros[0], zeros[1]))
        random.shuffle(unvisited_node_list)

        board_full = len(np.where(rollout_mat == 0)[0]) == 0
        if node.move is not None:
            have_winner = check_for_win(rollout_mat, node.move)
        else:
            have_winner = None
        cur_player = node.player


        while have_winner is None and (not board_full):
            cur_player *= -1
            move = unvisited_node_list.pop()
            rollout_mat[move[0]][move[1]] = cur_player

            # Check whether the board is full
            board_full = len(np.where(rollout_mat == 0)[0]) == 0
            have_winner = check_for_win(rollout_mat, move)

        if board_full:
            return 0
        else:
            return have_winner