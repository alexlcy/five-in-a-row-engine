from agent.base import Agent
import numpy as np
import time
import random
import math
from MCTS import monte_carlo_tree_search, Node
from encoder.base import get_encoder_by_name
from agent.helper import unvisited_nodes,find_children_priority
from fiveinarow import check_for_win


def pprint_tree(node, file=None, _prefix="", _last=True, level = 0, max_depth=1):
    _prefix += "   " if _last else "|  "
    print(_prefix, "`- " if _last else "|- ", {"Visit Value":np.round(node.visit_count,3),
                                               "Q value":np.round(node.q_value,3),
                                               "U value":np.round(node.u_value,3),
                                               "Prior": np.round(node.prior_value, 3),
                                               "Pos": node.move}, sep="", file=file)
    child_count = len(node.children)

    if level >= max_depth:
        return

    for i, child in enumerate(node.children.items()):
        _last = i == (child_count - 1)
        pprint_tree(child[1], file, _prefix, _last, level+1)


class AlphaGomokuNode(object):
    def __init__(self, mat, player, probability=1.0, move=None, parent=None):

        # Game atrritube
        self.game_state = mat
        self.player = player
        self.move = move

        # Tree attribute
        self.children = {}
        self.parent = parent

        # AlphaGo specific attribute
        self.visit_count = 0
        self.q_value = 0 # Value of action
        self.prior_value = probability
        self.u_value = probability # Value of utility (Probability/Number of visit)

    def select_child(self):
        return max(self.children.items(), key=lambda child: child[1].q_value + child[1].u_value)

    def expand_children(self, moves, probabilities):
        selected_child = np.argsort(probabilities)[::-1][:10]
        probabilities_selected = probabilities[selected_child]
        moves_selected = moves[selected_child]

        for move, prob in zip(moves_selected, probabilities_selected):
            game_state_tmp = np.copy(self.game_state)
            i = move // 8
            j = move % 8
            game_state_tmp[i][j] = self.player * -1

            if move not in self.children:
                self.children[(i,j)] = AlphaGomokuNode(game_state_tmp, self.player * -1, probability=prob, move=(i,j), parent=self)

    def update_values(self, leaf_value):
        if self.parent is not None:
            self.parent.update_values(leaf_value)

        self.visit_count += 1

        self.q_value += leaf_value / self.visit_count

        if self.parent is not None:
            c_u = 5
            self.u_value = c_u * np.sqrt(self.parent.visit_count) * \
                           self.prior_value / (1 + self.visit_count)



class AlphaGomokuAgent(Agent):

    def __init__(self, deep_policy_model, value_model,rollout_model, simulation_number, cur_player,
                 depth=5, lambda_value=0.9, rollout_limit=40):
        Agent.__init__(self)
        self._deep_policy_model = deep_policy_model
        self._value_model = value_model
        self._rollout_model = rollout_model
        self._rollout_encoder = get_encoder_by_name('allpattern', (8,8))
        self._deep_encoder = get_encoder_by_name('layer_20_encoder', (8,8))

        self.lambda_value = lambda_value
        self.cur_player = cur_player
        self.simulation_number = simulation_number
        self.depth = depth
        self.rollout_limit = rollout_limit

    def policy_probabilities(self, current_state, move):
        encoder = self._deep_encoder
        X_input = encoder.encode(current_state, self.cur_player, move)
        predict_prod = self._deep_policy_model.predict(X_input)

        # Finding out the legal moves
        zeros = np.where(current_state == 0)
        unvisited_node_list = list(zip(zeros[0], zeros[1]))
        legal_moves = np.array([i*8 + j for i, j in unvisited_node_list])

        # If there are no legal move left
        if len(legal_moves) == 0:
            return [], []

        # Getting the normalized probability of the legal move
        legal_outputs = predict_prod[0][legal_moves]
        normalized_outputs = legal_outputs / np.sum(legal_outputs)

        return legal_moves, normalized_outputs


    def select_move(self, mat, move):
        time_start = time.time()
        self.root = AlphaGomokuNode(mat, self.cur_player, move=move)

        for simulation in range(self.simulation_number):
            current_state = mat
            node = self.root

            for depth in range(self.depth):
                if not node.children:
                    if check_for_win(node.game_state, node.move) is not None:
                        break
                    moves, probabilities = self.policy_probabilities(current_state, move)
                    node.expand_children(moves, probabilities)

                move, node = node.select_child()
                current_state = np.copy(node.game_state)
                current_state[move[0]][move[1]] = node.player * -1

            current_state_input = self._deep_encoder.encode(current_state, node.player, move)
            value = self._value_model.predict(current_state_input)[0][0]

            if node.player == -1:
                value *= -1

            rollout = self.policy_rollout(current_state, move, node.player)
            weighted_value = (1 - self.lambda_value) * value + self.lambda_value * rollout
            node.update_values(weighted_value)

        move = max(self.root.children, key=lambda move: self.root.children.get(move).visit_count)
        mat[move[0]][move[1]] = self.cur_player

        pprint_tree(self.root)
        print("Total time spent", time.time()-time_start)

        return mat, move


    def policy_rollout(self, rollout_mat, move, player):
        zeros = np.where(rollout_mat == 0)
        unvisited_node_list = list(zip(zeros[0], zeros[1]))
        random.shuffle(unvisited_node_list)

        board_full = len(np.where(rollout_mat == 0)[0]) == 0
        if move is not None:
            have_winner = check_for_win(rollout_mat, move)
        else:
            have_winner = None
        cur_player = player

        while have_winner is None and (not board_full):
            cur_player *= -1
            move = unvisited_node_list.pop()
            rollout_mat[move[0]][move[1]] = cur_player

            # Fast model rollout
            # X_input = self._rollout_encoder.encode(rollout_mat,cur_player, move)
            # position_priority = list(np.argsort(self._rollout_model.predict(X_input))[0][::-1])
            # for position in position_priority:
            #     i = position // 8
            #     j = position % 8
            #     if rollout_mat[i][j] == 0:
            #         break
            # rollout_mat[i][j] = cur_player


            # Check whether the board is full
            board_full = len(np.where(rollout_mat == 0)[0]) == 0
            have_winner = check_for_win(rollout_mat, move)

        if board_full:
            return 0
        else:
            return have_winner


    def serialize(self, h5file):
        raise IOError("AlphaGoMCTS agent can\'t be serialized" +
                       "consider serializing the three underlying" +
                       "neural networks instad.")