from encoder.base import Encoder
import numpy as np

class Layer20Encoder(Encoder):
    def __init__(self, board_size):
        self.board_width, self.board_height = board_size
        self.num_planes = 1

    def num_points(self):
        return self.board_width * self.board_height

    def encode(self, mat, cur_player, last_move):
        # Flip the sign of the mat if the previous player is -1 (Current player is 1)
        if cur_player == -1:
            mat = np.negative(mat)
            mat = np.where(mat == -0., 0, mat)

        layer_my_move = np.array(mat > 0).astype(int)
        layer_oppo_move = np.array(mat < 0).astype(int)
        layer_valid_move = np.array(mat == 0).astype(int)
        layers_break_down = break_down_layers(mat)
        layers_open_moves = potential_open_moves(mat)
        layer_last_move = self.encode_point(last_move)
        layer_zeros = np.zeros((8,8))
        layer_ones = np.ones((8,8))
        final_X = np.dstack([layer_my_move, layer_oppo_move, layer_last_move, layer_valid_move]+ [layer_zeros] + [layer_ones] + layers_break_down + layers_open_moves)
        return np.expand_dims(final_X, axis = 0)

    def encode_point(self, move):
        temp_mat = np.zeros((self.board_width, self.board_height))
        if move is not None:
            temp_mat[move[0]][move[1]] = 1
        return temp_mat

def create(board_size):
    return Layer20Encoder(board_size)

def potential_open_moves(game_state):
    mat = np.copy(game_state)

    master_X_open_self = []
    master_X_open_oppo = []

    for search_num in np.arange(4,1,-1):
        X_open_self = np.zeros((8,8))
        X_open_oppo = np.zeros((8,8))
        X_record_self = np.zeros((8,8))
        X_record_oppo = np.zeros((8,8))

        # Did not consider open and close siutation
        m = 8
        n = 8
        for i in range(m):
            for j in range(m):

                if j + search_num+1 <= m:
                    sideway = mat[i][j:j+(search_num)]
                    if np.sum(sideway) ==search_num:
                        X_record_self[i][j:j+search_num] = 1

                        if mat[i][j-1] == 0:
                            X_open_self[i][j-1] = 1
                        if  mat[i][j+(search_num)] == 0:
                            X_open_self[i][j+(search_num)] = 1

                    if np.sum(sideway) == -search_num:
                        X_record_oppo[i][j:j+search_num] = 1

                        if mat[i][j-1] == 0:
                            X_open_oppo[i][j-1] = 1
                        if  mat[i][j+(search_num)] == 0:
                            X_open_oppo[i][j+(search_num)] = 1

                if i + search_num+1 <= m:
                    vert = mat[:,j][i:i+(search_num)]
                    if np.sum(vert) == search_num:
                        X_record_self[:,j][i:i+search_num] = 1

                        if mat[i-1][j] == 0:
                            X_open_self[i-1][j] = 1
                        if mat[i+(search_num)][j] == 0:
                            X_open_self[i+(search_num)][j] = 1

                    if np.sum(vert) == -search_num:
                        X_record_oppo[:,j][i:i+search_num] = 1

                        if mat[i-1][j] == 0:
                            X_open_oppo[i-1][j] = 1
                        if mat[i+(search_num)][j] == 0:
                            X_open_oppo[i+(search_num)][j] = 1


                if j + search_num+1 <= m and i + search_num+1 <= n:
                    diag = np.array([mat[i+x][j+y] for x in range(search_num) for y in range(search_num) if x == y])
                    if np.sum(diag) == search_num:
                        for k in range(search_num):
                            X_record_self[i+k][j+k] = 1


                        if mat[i-1][j-1] == 0:
                            X_open_self[i-1][j-1] = 1
                        if mat[i+(search_num)][j+(search_num)] == 0:
                            X_open_self[i+(search_num)][j+(search_num)] = 1

                    if np.sum(diag) == -search_num:
                        for k in range(search_num):
                            X_record_oppo[i+k][j+k] = 1

                        if mat[i-1][j-1] == 0:
                            X_open_oppo[i-1][j-1] = 1
                        if mat[i+(search_num)][j+(search_num)] == 0:
                            X_open_oppo[i+(search_num)][j+(search_num)] = 1

                if i + search_num+1 <= n and j - search_num >= 0:
                    diag = np.array([mat[i+x][j-y] for x in range(search_num) for y in range(search_num) if x == y])
                    if np.sum(diag) == search_num:
                        for k in range(search_num):
                            X_record_self[i+k][j-k] = 1

                        if j+1 < m and mat[i-1][j+1] == 0:
                            X_open_self[i-1][j-1] = 1
                        if  mat[i+(search_num)][j-(search_num)] == 0:
                            X_open_self[i+(search_num)][j-(search_num)] = 1
                    if np.sum(diag) == -search_num:
                        for k in range(search_num):
                            X_record_oppo[i+k][j-k] = 1

                        if j+1 < m and mat[i-1][j+1] == 0:
                            X_open_oppo[i-1][j+1] = 1
                        if mat[i+(search_num)][j-(search_num)] == 0:
                            X_open_oppo[i+(search_num)][j-(search_num)] = 1

        i_axis = list(np.where(X_record_self==1)[0]) + list(np.where(X_record_oppo==1)[0])
        j_axis = list(np.where(X_record_self==1)[1]) + list(np.where(X_record_oppo==1)[1])

        for i,j in zip(i_axis, j_axis):
            mat[i][j] = 0

        master_X_open_self.append(np.copy(X_open_self))
        master_X_open_oppo.append(np.copy(X_open_oppo))
    return master_X_open_self + master_X_open_oppo


def break_down_layers(game_state):
    mat = np.copy(game_state)

    master_X_open_self = []
    master_X_open_oppo = []

    for search_num in np.arange(4, 0, -1):
        X_open_self = np.zeros((8, 8))
        X_open_oppo = np.zeros((8, 8))

        # Did not consider open and close siutation
        m = 8
        n = 8
        for i in range(m):
            for j in range(m):
                if j + search_num + 1 <= m:
                    sideway = mat[i][j:j + (search_num)]
                    if np.sum(sideway) == search_num:
                        X_open_self[i][j:j + search_num] = 1
                    if np.sum(sideway) == -search_num:
                        X_open_oppo[i][j:j + search_num] = 1

                if i + search_num + 1 <= m:
                    vert = mat[:, j][i:i + (search_num)]
                    if np.sum(vert) == search_num:
                        X_open_self[:, j][i:i + search_num] = 1
                    if np.sum(vert) == -search_num:
                        X_open_oppo[:, j][i:i + search_num] = 1

                if j + search_num + 1 <= m and i + search_num + 1 <= n:
                    diag = np.array([mat[i + x][j + y] for x in range(search_num) for y in range(search_num) if x == y])
                    if np.sum(diag) == search_num:
                        for k in range(search_num):
                            X_open_self[i + k][j + k] = 1
                    if np.sum(diag) == -search_num:
                        for k in range(search_num):
                            X_open_oppo[i + k][j + k] = 1

                if j - search_num >= 0 and i + search_num + 1 <= n:
                    diag = np.array([mat[i + x][j - y] for x in range(search_num) for y in range(search_num) if x == y])
                    if np.sum(diag) == search_num:
                        for k in range(search_num):
                            X_open_self[i + k][j - k] = 1
                    if np.sum(diag) == -search_num:
                        for k in range(search_num):
                            X_open_oppo[i + k][j - k] = 1

        i_axis = list(np.where(X_open_self == 1)[0]) + list(np.where(X_open_oppo == 1)[0])
        j_axis = list(np.where(X_open_self == 1)[1]) + list(np.where(X_open_oppo == 1)[1])

        for i, j in zip(i_axis, j_axis):
            mat[i][j] = 0

        master_X_open_self.append(np.copy(X_open_self))
        master_X_open_oppo.append(np.copy(X_open_oppo))
    return master_X_open_self + master_X_open_oppo