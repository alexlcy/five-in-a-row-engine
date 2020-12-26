import numpy as np
from agent.mctsAgent import MCTSAgent
from agent.policyAgent import PolicyAgent

import datetime
import h5py
from scipy.stats import binom_test
import tensorflow as tf
from rl.experience import ExperienceCollector, combine_experience
import time
from agent.actorCriticAgent import ActorCriticAgent
from multiprocessing import Process, Value, Lock
from encoder.base import get_encoder_by_name
from fiveinarow import check_for_done, draw_board, render, check_for_done


class SimulationEngine:
    def __init__(self, bot_white, bot_black):
        self.bot_1 = bot_white
        self.bot_2 = bot_black
        self.player_bot_dict = {1: self.bot_1, -1: self.bot_2}
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


    def self_play_RL(self, num_games, file_path = "experience_data/"):
        time_start = time.time()
        white_win = 0
        draw = 0
        cur_player = 1

        collector1 = ExperienceCollector()
        collector2 = ExperienceCollector()

        self.bot_1.set_collector(collector1)
        self.bot_2.set_collector(collector2)

        self.bot_1.set_temperature(0.005)
        self.bot_2.set_temperature(0.005)


        for i in range(num_games):
            collector1.begin_episode()
            collector2.begin_episode()

            winner = self.simulate_game(cur_player=cur_player)

            if winner == 1:
                collector1.complete_episode(reward=1)
                collector2.complete_episode(reward=-1)
                white_win += 1
            elif winner == -1:
                collector1.complete_episode(reward=-1)
                collector2.complete_episode(reward=1)
            else:
                collector1.complete_episode(reward=0)
                collector2.complete_episode(reward=0)
                draw +=1

            cur_player *= -1

        experience = combine_experience([collector1,collector2])

        date_string = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        with h5py.File(file_path + "experiment_" +date_string , 'w') as experience_outf:
            experience.serialize(experience_outf)

        time_spent = time.time() - time_start
        print(f"Completed self play for {num_games} game. Total time: {time_spent}. Rate {time_spent/num_games}")
        print(f"White win num:{white_win}; Black win num: {num_games-white_win-draw}")


    def simulate_game(self, cur_player=1):
        game_state = np.zeros((self.M, self.M))
        cur_move = None

        while check_for_done(game_state)[0] is False:
            game_state, cur_move = self.player_bot_dict[cur_player].select_move(game_state, cur_move)
            cur_player *= -1

        print("Done for one game")
        return check_for_done(game_state)[1]


    def test_RL_Agents(self, num_games):
        time_start = time.time()
        wins = 0
        losses = 0
        draws = 0
        start_player = 1
        while wins + losses < num_games:
            winner = self.simulate_game(cur_player=start_player)
            start_player *= -1
            if winner == 1:
                wins += 1
            elif winner == -1:
                losses += 1
            else:
                draws +=1

        time_spent = time.time() - time_start
        print('Agent 1 (Testing Agent)')
        print('Total Win: %d' % (wins))
        print('Total Draws: %d' % (draws))
        print('Total Losses: %d' % (losses))
        print(f'Total Win percentage: {wins/(wins+losses)}')
        print(f"Total time: {time_spent}")
        print(f"p value {binom_test(wins, wins+losses, 0.5)}")

if __name__ == "__main__":

    """Reinforcement learning self play"""
    # model = tf.keras.models.load_model('saved_model/layer_20_model')
    # encoder = get_encoder_by_name('layer_20_encoder', (8,8))
    # bot_white = PolicyAgent(model, encoder, 1)
    # bot_black = PolicyAgent(model, encoder, -1)
    # bot_black.set_temperature(0.005)
    # bot_white.set_temperature(0.005)
    # simulation = SimulationEngine(bot_white, bot_black)
    # simulation.self_play_RL(num_games=2000)

    # model = tf.keras.models.load_model('saved_model/actor_critic_net')
    # encoder = get_encoder_by_name('layer_20_encoder', (8,8))
    # bot_white = ActorCriticAgent(model, encoder, 1)
    # bot_black = ActorCriticAgent(model, encoder, -1)
    # simulation = SimulationEngine(bot_white, bot_black)
    # simulation.self_play_RL(10)

    # See whether we can parallel in the future
    # num_cpu = 10
    # workers = []
    # for i in range(num_cpu):
    #     model = tf.keras.models.load_model('saved_model/layer_20_model')
    #     encoder = get_encoder_by_name('layer_20_encoder', (8, 8))
    #     bot_white = PolicyAgent(model, encoder, 1)
    #     bot_black = PolicyAgent(model, encoder, -1)
    #     simulation = SimulationEngine(bot_white, bot_black)
    #     worker = Process(simulation.self_play_RL,
    #                      args=(10))
    #     worker.start()
    #     workers.append(worker)
    # print('Waiting for workers...')
    # for worker in workers:
    #     worker.join()



    """Monte Carlo Geneerate Data"""
    # SIMULATION_NUMBER = 15000
    # TEMPERATURE = 0.5
    # encoder = get_encoder_by_name('oneplane', 8)
    # bot_white = MCTSAgent(simulation_number=SIMULATION_NUMBER,
    #                       temperature=TEMPERATURE,
    #                       cur_player=1)
    # bot_black = MCTSAgent(simulation_number=SIMULATION_NUMBER,
    #                       temperature=TEMPERATURE,
    #                       cur_player=-1)
    # simulation = SimulationEngine(bot_white, bot_black)
    # simulation.self_play_monte_carlo(encoder)

    # """Agent self play"""
    model = tf.keras.models.load_model('saved_model/pg_model_V2_00002_rate')
    encoder = get_encoder_by_name('layer_20_encoder', (8,8))
    bot_white = PolicyAgent(model, encoder, 1)

    model = tf.keras.models.load_model('saved_model/layer_20_model')
    bot_black = PolicyAgent(model, encoder, -1)

    bot_black.set_temperature(0)
    bot_white.set_temperature(0)

    simulation = SimulationEngine(bot_white, bot_black)
    simulation.test_RL_Agents(100)