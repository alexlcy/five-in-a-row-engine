from agent.base import Agent
from encoder.base import get_encoder_by_name
import tensorflow as tf
import numpy as np
import h5py
from tensorflow.keras.optimizers import SGD

encoder = get_encoder_by_name('allpattern', 8)


class PolicyAgent(Agent):
    def __init__(self,model,encoder, cur_player):
        Agent.__init__(self)
        self._model = model
        self._encoder = encoder
        self.cur_player = cur_player

    def serialize(self, h5file):
        h5file.create_group('encoder')
        h5file['encoder'].attrs['name'] = self.encoder.name()
        h5file['encoder'].attrs['board_width'] = self.encoder.board_width
        h5file['encoder'].attrs['board_height'] = self.encoder.board_height
        h5file.create_group('model')
        self._model.save(h5file)

    def load_policy_agent(h5file, cur_player):
        model = tf.keras.models.load_model(h5file['model'])
        encoder_name = h5file['encoder'].attrs['name']
        board_width = h5file['encoder'].attrs['board_width']
        board_height = h5file['encoder'].attrs['board_height']
        encoder = get_encoder_by_name(encoder_name,(board_width, board_height))
        return PolicyAgent(model, encoder, cur_player)

    def clip_probs(self, original_probs):
        min_p = 1e-5
        max_p = 1 - min_p
        clipped_probs = np.clip(original_probs, min_p, max_p)
        clipped_probs = clipped_probs / np.sum(clipped_probs)
        return clipped_probs

    def select_move(self, mat, move):
        X_input = encoder.encode(mat, self.cur_player, move)
        X_input = tf.convert_to_tensor(X_input)
        move_probs = self._model.predict(X_input)
        move_probs = self.clip_probs(move_probs)
        num_moves = 8*8
        candidates = np.arange(num_moves)
        ranked_moves = np.random.choice(candidates,
                                        num_moves,
                                        replace=False,
                                        p=move_probs[0])
        for position in ranked_moves:
            i = position // 8
            j = position % 8
            if mat[i][j] == 0:
                break

        if self.collector is not None:
            temp_mat = np.zeros((8,8))
            if move is not None:
                temp_mat[move[0]][move[1]] = 1

            self.collector.record_decision(
                    state=mat,
                    action=temp_mat
            )

        mat[i][j] = self.cur_player

        return mat, (i,j)

    def train(self, experience, lr, clipnorm, batch_size):
        self._model.compile(loss = 'categorical_crossentropy',optimizer=SGD(lr=lr, clipnorm=clipnorm))
        target_vectors = prepare_experience_data(
            experience,
            self._encoder.board_width,
            self._encoder.board_height)
        self._model.fit(
            experience.states, target_vectors,
            batch_size=batch_size,
            epochs=1)

    def set_collector(self, collector):
        self.collector = collector



def prepare_experience_data(experience, board_width, board_height):
    experience_size = experience.actions.shape[0]
    target_vectors = np.zeros((experience_size, board_width * board_height))
    for i in range(experience_size):
        action = experience.actions[i]
        reward = experience.rewards[i]
        target_vectors[i][action] = reward
    return target_vectors