from agent.base import Agent
from encoder.base import get_encoder_by_name
import tensorflow as tf
import numpy as np
import datetime
import h5py
from tensorflow.keras.optimizers import SGD



class ActorCriticAgent(Agent):
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

    def load_actor_critic_agent(h5file, cur_player):
        model = tf.keras.models.load_model(h5file['model'])
        encoder_name = h5file['encoder'].attrs['name']
        board_width = h5file['encoder'].attrs['board_width']
        board_height = h5file['encoder'].attrs['board_height']
        encoder = get_encoder_by_name(encoder_name,(board_width, board_height))
        return ActorCriticAgent(model, encoder, cur_player)

    def clip_probs(self, original_probs):
        min_p = 0.01
        max_p = 1 - min_p
        clipped_probs = np.clip(original_probs, min_p, max_p)
        clipped_probs = clipped_probs / np.sum(clipped_probs)
        return clipped_probs

    def select_move(self, mat, move):
        X_input = self._encoder.encode(mat, self.cur_player, move)
        X_input = tf.convert_to_tensor(X_input)
        move_probs, values = self._model.predict(X_input)
        print(mat)
        print(values)
        estimated_value = values[0]

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
                    raw_state=mat,
                    state=X_input,
                    action=temp_mat,
                    estimated_value=estimated_value
            )

        mat[i][j] = self.cur_player

        return mat, (i,j)

    def train(self, experience, lr=0.0000001, batch_size=512):
        self._model.model.compile(loss = ['categorical_crossentropy', 'mse'],
                                  optimizer='adam',
                                  loss_weights=[1.0, 0.5])

        n = experience.states.shape[0]
        num_moves = self.encoder.num_points()

        # Prepare the raw dataset
        feature_vector = np.squeeze(experience.states, axis=1)
        policy_target = np.zeros((n, num_moves))
        value_target = np.zeros((n,))

        for i in range(n):
            # Collect policy target of that round (Action position = estimated value of board [Advantage])
            action = experience.actions[i]
            action_pos = np.where(action == 1)
            policy_target[i][(action_pos[0][0] * 8) + action_pos[1][0]] = experience.advantages[i]

            # Collect reward info of the round (Actual win or loss in the round)
            reward = experience.rewards[i]
            value_target[i] = reward

        self.model.fit(
            feature_vector,
            [policy_target, value_target],
            batch_size=batch_size,
            epochs=1)

    def set_collector(self, collector):
        self.collector = collector

    def save(self, path = "saved_model/"):
        date_string = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self._model.save(path + "pg_model_actorcritic_" + date_string)

def prepare_experience_data(experience, board_width, board_height):
    experience_size = experience.actions.shape[0]
    target_vectors = np.zeros((experience_size, board_width * board_height))
    for i in range(experience_size):
        action = experience.actions[i]
        reward = experience.rewards[i]
        action_pos = np.where(action == 1)
        if np.sum(action) == 0:
            continue
        else:
            target_vectors[i][(action_pos[0][0]*8) + action_pos[1][0]] = reward
    return target_vectors.reshape(target_vectors.shape[0], 64)