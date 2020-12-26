from agent.base import Agent
from encoder.base import get_encoder_by_name
import tensorflow as tf
import numpy as np
import datetime
import h5py
import random
from tensorflow.keras.optimizers import SGD


class PolicyAgent(Agent):
    def __init__(self,model,encoder, cur_player):
        Agent.__init__(self)
        self._model = model
        self._encoder = encoder
        self.cur_player = cur_player
        self.collector = None
        self._temperature = 0.0

    def set_temperature(self, temperature):
        self._temperature = temperature

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
        X_input_np = self._encoder.encode(mat, self.cur_player, move)
        num_moves = self._encoder.board_width * self._encoder.board_height

        # Use random strategy based on the temperature threshold
        if np.random.random() < self._temperature:
            move_probs = np.ones(num_moves) / num_moves
        # Otherwise, use the deep learning prediction
        else:
            X_input = tf.convert_to_tensor(X_input_np)
            move_probs = self._model.predict(X_input)[0]

        # Clip probability
        move_probs = self.clip_probs(move_probs)
        candidates = np.arange(num_moves)
        ranked_moves = np.random.choice(candidates,num_moves,replace=False,p=move_probs)

        # Try to apply the move to the positions to check out whether it is a legal move
        for position in ranked_moves:
            i = position // 8
            j = position % 8

            if mat[i][j] == 0:
                break

        # Collect experience while doing self-play
        if self.collector is not None:
            temp_mat = np.zeros((8,8))
            temp_mat[i][j] = 1

            # record decision
            self.collector.record_decision(
                    raw_state=np.copy(mat),
                    state=X_input_np,
                    action=temp_mat)

        # Place the stone to the board after recording
        mat[i][j] = self.cur_player

        # Return the board and the move
        return mat, (i,j)

    def train(self, experience, lr=0.000000001, clipnorm=1.0, batch_size=4096):
        self._model.compile(loss='categorical_crossentropy',
                            optimizer=SGD(lr=lr, clipnorm=clipnorm))

        target_vectors = prepare_experience_data(
            experience,
            self._encoder.board_width,
            self._encoder.board_height)

        feature_vector = np.squeeze(experience.states, axis=1)

        self._model.fit(
            feature_vector,
            target_vectors,
            batch_size=batch_size,
            epochs=1)

    def set_collector(self, collector):
        self.collector = collector

    def save(self, version, path = "saved_model/"):
        self._model.save(path + f"pg_model_V{version}")



def prepare_experience_data(experience, board_width, board_height):
    experience_size = experience.actions.shape[0]
    target_vectors = np.zeros((experience_size, board_width * board_height))

    # Iterate through the saved action and reward. Convert them into policy gradient model input
    for i in range(experience_size):
        action = experience.actions[i]
        reward = experience.rewards[i]
        action_pos = np.where(action == 1)

        if np.sum(action) == 0:
            continue
        else:
            # Set the target value of the action position to be reward
            target_vectors[i][(action_pos[0][0]*8) + action_pos[1][0]] = reward

    return target_vectors