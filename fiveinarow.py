#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 20:36:53 2019

@author: root
"""
import pygame
from numba import jit

M=8
import numpy as np

def update_by_man(event, mat):
    """
    This function detects the mouse click on the game window. Update the state matrix of the game.
    input:
        event:pygame event, which are either quit or mouse click)
        mat: 2D matrix represents the state of the game
    output:
        mat: updated matrix
    """
    global M
    done=False
    if event.type==pygame.QUIT:
        done=True
    if event.type==pygame.MOUSEBUTTONDOWN:
        (x,y)=event.pos
        row = round((y - 40) / 40)
        col = round((x - 40) / 40)
        mat[row][col]=1
    return mat, done


def draw_board(screen):
    """
    This function draws the board with lines.
    input: game windows
    output: none
    """
    global M
    d=int(560/(M-1))
    black_color = [0, 0, 0]
    board_color = [241, 196, 15]
    screen.fill(board_color)
    for h in range(0, M):
        pygame.draw.line(screen, black_color,[40, h * d+40], [600, 40+h * d], 1)
        pygame.draw.line(screen, black_color, [40+d*h, 40], [40+d*h, 600], 1)

def draw_stone(screen, mat):
    """
    This functions draws the stones according to the mat. It draws a black circle for matrix element 1(human),
    it draws a white circle for matrix element -1 (computer)
    input:
        screen: game window, onto which the stones are drawn
        mat: 2D matrix representing the game state
    output:
        none
    """
    black_color = [0, 0, 0]
    white_color = [255, 255, 255]
    M=len(mat)
    d=int(560/(M-1))
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if mat[i][j]==1:
                pos = [40+d * j, 40+d* i ]
                pygame.draw.circle(screen, black_color, pos, 18,0)
            elif mat[i][j]==-1:
                pos = [40+d* j , 40+d * i]
                pygame.draw.circle(screen, white_color, pos, 18,0)



def render(screen, mat):
    """
    Draw the updated game with lines and stones using function draw_board and draw_stone
    input:
        screen: game window, onto which the stones are drawn
        mat: 2D matrix representing the game state
    output:
        none
    """
    draw_board(screen)
    draw_stone(screen, mat)
    pygame.display.update()


def check_for_done(mat):
    """
    please write your own code testing if the game is over. Return a boolean variable done. If one of the players wins
    or the tie happens, return True. Otherwise return False. Print a message about the result of the game.
    input:
        2D matrix representing the state of the game
    output:
        none
    """
    m, n = mat.shape
    target1 = [1, 1, 1, 1, 1]
    target2 = [-1, -1, -1, -1, -1]
    if len(np.where(mat == 0)[0]) == 0:
        return True, 0.5

    for i in range(m):
        for j in range(n):
            if j + 5 <= m:
                sideway = mat[i][j:j + 5]
                if (sideway == target1).all():
                    return True, 1
                if (sideway == target2).all():
                    return True, -1
            if i + 5 <= m:
                vert = mat[:, j][i:i + 5]
                if (vert == target1).all():
                    return True, 1
                if (vert == target2).all():
                    return True, -1
            if j + 5 <= m and i + 5 <= n:
                diag = [mat[i + x][j + y] for x in range(5) for y in range(5) if x == y]
                if diag == target1:
                    return True, 1
                if diag == target2:
                    return True, -1
            if j - 4 >= 0 and i + 5 <= n:
                diag = [mat[i + x][j - y] for x in range(5) for y in range(5) if x == y]
                if diag == target1:
                    return True, 1
                if diag == target2:
                    return True, -1
    return False, 0

@jit(nopython=True)
def check_for_win(mat, move):
    n = mat.shape[0]
    i = move[0]
    j = move[1]
    player = mat[i][j]

    if len(np.where(mat==0)[0]) == 0:
        return 0

    offset = {'h': (0, 1), 'v': (1, 0), 'd': (1, 1), 'ad': (-1, 1)}
    for os in offset.values():
        i1 = i + os[0]
        j1 = j + os[1]
        i2 = i + -os[0]
        j2 = j + -os[1]
        counting = 1
        while i1 >= 0 and j1 >= 0 and i1 < n and j1 < n:
            if mat[i1][j1] == player:
                counting += 1
                i1 += os[0]
                j1 += os[1]
            else:
                break
        while i2 >= 0 and j2 >= 0 and i2 < n and j2 < n:
            if mat[i2][j2] == player:
                counting += 1
                i2 -= os[0]
                j2 -= os[1]
            else:
                break
        if counting >= 5:
            return player


def find_children_priority(state, subset, player):
    ''' this is to select which children to expand, different scores are added to each move
        if the move leads to different scenarios:
        1. 5 (inf)
        2. break 5 (20000)
        3. open 4 (10000)
        4. close 4 and open 3 (5000)
        5. break open 4 (4000)
        6. break (close 4 and open 3) (3000)
        7. open 3 x 2 (2000)
        8. break (open 3 x 2) (1000)
        9. open 2 x 3 (300)
        10. open 2 x 2 (200)
        11. open 2 (100)
        '''
    n = state.shape[0]
    scores = {}
    offset = {'h': (0, 1), 'v': (1, 0), 'd': (1, 1), 'ad': (-1, 1)}
    for pos in subset:
        i = pos[0]
        j = pos[1]
        scores[(i, j)] = 0
        closed_4 = False
        break_closed_4 = False
        for os in offset.values():
            # Linking Count
            i1 = i + os[0]
            j1 = j + os[1]
            i2 = i - os[0]
            j2 = j - os[1]
            linking = 1
            is_open = True
            while i1 >= 0 and j1 >= 0 and i1 < n and j1 < n:
                if state[i1][j1] == player:
                    linking += 1
                    i1 += os[0]
                    j1 += os[1]
                elif state[i1][j1] != 0:
                    is_open = False
                    break
                else:
                    break
            while i2 >= 0 and j2 >= 0 and i2 < n and j2 < n:
                if state[i2][j2] == player:
                    linking += 1
                    i2 -= os[0]
                    j2 -= os[1]
                elif state[i2][j2] != 0:
                    is_open = False
                    break
                else:
                    break
            if i1 < 0 or j1 < 0 or i1 == n or j1 == n or i2 < 0 or j2 < 0 or i2 == n or j2 == n:
                is_open = False
            if linking >= 5:
                scores[(i, j)] = float('inf')
                break
            elif linking == 4 and is_open:
                scores[(i, j)] += 10000
            elif linking == 4:
                closed_4 = True
            elif linking == 3 and closed_4:
                scores[(i, j)] += 5000
            elif linking == 3 and is_open:
                scores[(i, j)] += 1000
            elif linking == 2 and is_open:
                scores[(i, j)] += 100
            # Breaking Count
            i1 = i + os[0]
            j1 = j + os[1]
            i2 = i - os[0]
            j2 = j - os[1]
            breaking = 1
            is_open = True
            while i1 >= 0 and j1 >= 0 and i1 < n and j1 < n:
                if state[i1][j1] == -player:
                    breaking += 1
                    i1 += os[0]
                    j1 += os[1]
                elif state[i1][j1] != 0:
                    is_open = False
                    break
                else:
                    break
            while i2 >= 0 and j2 >= 0 and i2 < n and j2 < n:
                if state[i2][j2] == -player:
                    breaking += 1
                    i2 -= os[0]
                    j2 -= os[1]
                elif state[i2][j2] != 0:
                    is_open = False
                    break
                else:
                    break
            if i1 < 0 or j1 < 0 or i1 == n or j1 == n or i2 < 0 or j2 < 0 or i2 == n or j2 == n:
                is_open = False
            if breaking >= 5:
                scores[(i, j)] = 20000
                break
            elif breaking == 4 and is_open:
                scores[(i, j)] += 4000
            elif breaking == 4:
                break_closed_4 = True
            elif breaking == 3 and break_closed_4:
                scores[(i, j)] += 3000
            elif breaking == 3 and is_open:
                scores[(i, j)] += 500
    sorted_score = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [i[0] for i in sorted_score[:5]]
