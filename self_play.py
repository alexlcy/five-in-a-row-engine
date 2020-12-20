import numpy as np
from agent.mctsAgent import MCTSAgent
from agent.policyAgent import PolicyAgent
import datetime
import h5py
import tensorflow as tf
from rl.experience import ExperienceCollector, combine_experience
import time
from encoder.base import get_encoder_by_name
from fiveinarow import check_for_done, draw_board, render, check_for_done


class SimulationEngine:
    def __init__(self, bot_white, bot_black):
        self.bot_white = bot_white
        self.bot_black = bot_black
        self.player_bot_dict = {1: bot_white, -1: bot_black}
        self.M = 8

    def pygame_self_play(self):
        import pygame
        pygame.init()
        screen = pygame.display.set_mode((640, 640))
        draw_board(screen)
        pygame.display.update()

        while True:
            step_num = 0
            mat = np.zeros((self.M, self.M))
            cur_move = None
            cur_player = 1
            while check_for_done(mat)[0] is False:
                render(screen, mat)
                mat, cur_move = self.player_bot_dict[cur_player].select_move(mat, cur_move)
                cur_player *= -1
                step_num +=1

                print('Step Number', step_num)
                print(mat)
                print("--------------------------")
                render(screen, mat)

        pygame.quit()

    def self_play_monte_carlo(self, encoder, saving_perod = 50, file_path = 'game_data/'):
        print("self play started")
        date_string = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        start_time = time.time()
        file_length = 0
        master_mat = []
        master_move = []
        master_result = []

        while True:
            step_num = 0
            mat = np.zeros((self.M, self.M))
            cur_move = None
            cur_player = 1

            while check_for_done(mat)[0] is False:

                # Saving the mat before the game
                master_mat.append(encoder.encode(mat, cur_player))

                # Use the agent to take move
                mat, cur_move = self.player_bot_dict[cur_player].select_move(mat, cur_move)

                # Saving the move information
                master_move.append(encoder.encode_point(cur_move))
                cur_player *= -1
                step_num += 1
            master_result += [check_for_done(mat)[1] for _ in range(step_num)]

            time_diff = time.time() - start_time
            print(f'Done for one game with {step_num} moves.Time Spent :{time_diff}.Rate: {time_diff/len(master_move)}')

            if len(master_move) > file_length + saving_perod:
                master_mat_save = np.array(master_mat)
                master_move_save = np.array(master_move)
                master_result_save = np.array(master_result)
                time_diff = time.time() - start_time
                print(f"Saving file for {len(master_move)} steps now. Time Spent :{time_diff}. Rate: {time_diff/len(master_move)}")


                print(master_mat_save.shape)
                print(master_move_save.shape)
                print(master_result_save.shape)

                np.save(file_path + f"master_mat_rule_{date_string}.npy", master_mat_save)
                np.save(file_path + f"master_move_rule_{date_string}.npy", master_move_save)
                np.save(file_path + f"master_result_rule_{date_string}.npy", master_result_save)
                file_length = len(master_move)

    def simulate_game(self):
        game_state = np.zeros((self.M, self.M))
        cur_player = 1
        cur_move = None

        while check_for_done(game_state)[0] is False:
            game_state, cur_move = self.player_bot_dict[cur_player].select_move(game_state, cur_move)
            cur_player *= -1

        return check_for_done(game_state)[1]

    def self_play_RL(self, num_games, file_path = "experience_data"):
        time_start = time.time()
        white_win = 0

        collector1 = ExperienceCollector()
        collector2 = ExperienceCollector()

        self.bot_white.set_collector(collector1)
        self.bot_black.set_collector(collector2)


        for i in range(num_games):
            collector1.begin_episode()
            collector2.begin_episode()

            winner = self.simulate_game()
            if winner == 1:
                collector1.complete_episode(reward=1)
                collector2.complete_episode(reward=-1)
                white_win +=1
            elif winner == -1:
                collector2.complete_episode(reward=1)
                collector1.complete_episode(reward=-1)
            else:
                collector2.complete_episode(reward=0.2)
                collector1.complete_episode(reward=0.2)

        experience = combine_experience([collector1,collector2])

        date_string = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        with h5py.File(file_path + "experiment_" +date_string , 'w') as experience_outf:
            experience.serialize(experience_outf)

        time_spent = time.time()-time_start
        print(f"Completed self play for {num_games} game. Total time: {time_spent}. Rate {time_spent/num_games}")
        print(f"White win num:{white_win}; Black win num: {num_games-white_win}")


if __name__ == "__main__":
    model = tf.keras.models.load_model('saved_model/allpattern_model')
    encoder = get_encoder_by_name('allpattern', (8,8))
    # bot_white = PolicyAgent(model, encoder, 1)
    # bot_black = PolicyAgent(model, encoder, -1)

    SIMULATION_NUMBER = 15000
    TEMPERATURE = 0.5
    encoder = get_encoder_by_name('oneplane', 8)
    bot_white = MCTSAgent(simulation_number=SIMULATION_NUMBER,
                          temperature=TEMPERATURE,
                          cur_player=1)
    bot_black = MCTSAgent(simulation_number=SIMULATION_NUMBER,
                          temperature=TEMPERATURE,
                          cur_player=-1)
    simulation = SimulationEngine(bot_white, bot_black)
    simulation.self_play_monte_carlo(encoder)

    # simulation = SimulationEngine(bot_white, bot_black)
    # simulation.self_play_RL(20000)