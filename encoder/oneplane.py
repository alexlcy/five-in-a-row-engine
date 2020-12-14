from encoder.base import Encoder
import numpy as np

class OnePlaneEncoder(Encoder):
    def __init__(self, board_size):
        self.board_width, self.board_height = board_size
        self.num_planes = 1

    def encode(self, mat, cur_player):
        # Flip the sign of the mat if the previous player is -1 (Current player is 1)
        if cur_player == 1:
            mat = np.negative(mat)
            mat = np.where(mat == -0., 0, mat)
        return mat

    def encode_point(self, move):
        temp_mat = np.zeros((self.board_width, self.board_height))
        temp_mat[move[0]][move[1]] = 1
        return temp_mat


def create(board_size):
    return OnePlaneEncoder(board_size)