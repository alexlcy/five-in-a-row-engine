import numpy as np
import pygame
from fiveinarow import draw_board, render, check_for_done
from agent import randomAgent, mctsAgent, deeplearningAgent
from MCTS import Node, update_root, monte_carlo_tree_search
from multiprocessing import Process
import multiprocessing

def update_by_pc(mat, move):
    """
    This is the core of the game. Write your code to give the computer the intelligence to play a Five-in-a-Row game
    with a human
    input:
        2D matrix representing the state of the game.
    output:
        2D matrix representing the updated state of the game.
    """
    #bots = randomAgent.RandomAgent()
    #bots = mctsAgent.MCTSAgent(simulation_number=15000, temperature=0.5, cur_player=1)
    bots = deeplearningAgent.DeepLearningAgent(cur_player=-1)
    mat, move = bots.select_move(mat, move)
    return mat

def main(M = 8):
    pygame.init()
    screen=pygame.display.set_mode((640,640))
    pygame.display.set_caption('Five-in-a-Row')
    done=False
    mat=np.zeros((M,M))
    mat[4][4] = -1
    d=int(560/(M-1))
    draw_board(screen)
    pygame.display.update()
    result = None

    while not done:
        for event in pygame.event.get():
            if event.type==pygame.QUIT:
                done=True
            render(screen, mat)
            if event.type == pygame.MOUSEBUTTONDOWN:
                (x, y) = event.pos
                row = round((y - 40) / d)
                col = round((x - 40) / d)
                mat[row][col] = 1
                render(screen, mat)
                done, result = check_for_done(mat)
                if done:
                    break
                else:
                    mat = update_by_pc(mat, (row, col))
                done, result = check_for_done(mat)
                if done:
                    break
    pygame.quit()
    print("winner is:", result)

if __name__ == '__main__':
    main()
