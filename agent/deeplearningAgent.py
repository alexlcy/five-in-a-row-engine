from agent.base import Agent
from encoder.base import get_encoder_by_name
import tensorflow as tf
import numpy as np

encoder = get_encoder_by_name('allpattern', 8)

class DeepLearningAgent(Agent):
    def __init__(self, cur_player):
        Agent.__init__(self)
        self.cur_player = cur_player
        self.model = model_loaded = tf.keras.models.load_model('saved_model/allpattern_model')

    def select_move(self, mat, move):
        X_input = encoder.encode(mat, self.cur_player, move)

        position_priority = list(np.argsort(self.model.predict(X_input))[0][::-1])
        for position in position_priority:
            i = position // 8
            j = position % 8
            if mat[i][j] == 0:
                break

        mat[i][j] = self.cur_player

        return mat, (i,j)


