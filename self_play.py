import numpy as np
from agent.mctsAgent import MCTSAgent
import datetime
import time
from encoder.base import get_encoder_by_name
from fiveinarow import check_for_done, draw_board, render, check_for_done

SIMULATION_NUMBER = 10000
TEMPERATURE = 0.5
encoder = get_encoder_by_name('oneplane', 8)

def self_play(M = 8):
    print("self play started")
    date_string = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    start_time = time.time()
    mat = np.zeros((M, M))
    cur_move = None
    cur_player = 1
    file_length = 0
    debug = True
    master_mat = []
    master_move = []

    bot_white = MCTSAgent(simulation_number=SIMULATION_NUMBER,
                          temperature=TEMPERATURE,
                          cur_player=1)
    bot_black = MCTSAgent(simulation_number=SIMULATION_NUMBER,
                          temperature=TEMPERATURE,
                          cur_player=-1)

    player_bot_dict = {1: bot_white, -1: bot_black}

    # if debug == True:
    #     pygame.init()
    #     screen = pygame.display.set_mode((640, 640))
    #     draw_board(screen)
    #     pygame.display.update()

    while True:
        step_num = 0
        while check_for_done(mat)[0] is False:
            # if debug == True and pygame.event.get() is not None:
            #     render(screen, mat)

            # Saving the mat before the game
            master_mat.append(encoder.encode(mat, cur_player))

            # Use the agent to take move
            mat, cur_move = player_bot_dict[cur_player].select_move(mat, cur_move)

            # Saving the move information
            master_move.append(encoder.encode_point(cur_move))
            cur_player *= -1
            step_num +=1
        time_diff = time.time() - start_time
        print(f'Done for one game with {step_num} moves.Time Spent :{time_diff}.Rate: {time_diff/len(master_move)}')
        print(len(master_move))

        if len(master_move) > file_length + 100:
            master_mat = np.concatenate(master_mat)
            master_move = np.concatenate(master_move)

            time_diff = time.time() - start_time
            print(f"Saving file for {len(master_move)} steps now. Time Spent :{time_diff}. Rate: {time_diff/len(master_move)}")

            np.save(f"game_data/master_mat_{date_string}.npy", master_mat)
            np.save(f"game_data/master_move_{date_string}.npy", master_move)
            file_length = len(master_move)

    #         if debug == True:
    #             print('Step Number', step_num)
    #             print(mat)
    #             print("--------------------------")
    #             render(screen, mat)
    #
    # if debug == True:
    #     pygame.quit()

self_play()