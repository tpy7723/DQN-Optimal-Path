# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import numpy as np
import Qmaze_
import maze_
import qtrain_
import time

maze = maze_.total_maze
env = maze_.env

def play_game(model, qmaze, rat_cell):
    global win_counter, lose_counter

    qmaze.reset(rat_cell)  # 입력 받은 위치로 로봇을 맵에 초기화함
    envstate = qmaze.observe()  # 현재 스테이트를 받아옴

    while True:
        prev_envstate = envstate  # 스테이트를 전환
        # print("envstate: " , envstate)
        # get next action
        q = model.predict(prev_envstate)  # 스테이트를 넣고 q 배열이 나옴
        # print("q: ", q)
        action = np.argmax(q[0])  # 큰 Q 벨류가 액션

        # apply action, get rewards and new state
        env.step(action)  # 액션 취함
        env.render()

        time.sleep(0.1)

        envstate, reward, game_status = qmaze.act(action)

        if game_status == 'win':  # 목적지에 도착하면
            print("win\n")  # win
            win_counter += 1
            env.countWin(win_counter)
            return True  # true

        elif game_status == 'lose':  # 갈 곳이 없으면
            print("lose\n")  # lose
            lose_counter += 1
            env.countLose(lose_counter)
            return False  # false


def trainMat(maze, env, model):
    global win_counter, lose_counter
    temp = maze

    env.changeMap(temp)

    # 기존 것에 더 학습
    qtrain_.qtrain(model, temp, epochs=1000, max_memory=800 * maze.size, data_size=32, weights_file="model.h5")
    model.save('model.h5')
    env.countWin(0)
    env.countLose(0)
    win_counter = 0
    lose_counter = 0

def trainMat_new(maze, env, model):
    global win_counter, lose_counter
    temp = maze

    env.changeMap(temp)

    # 기존 것에 더 학습
    qtrain_.qtrain(model, temp, epochs=1000, max_memory=800 * maze.size, data_size=32)
    model.save('model.h5')
    env.countWin(0)
    env.countLose(0)
    win_counter = 0
    lose_counter = 0


def confirmResult(maze, env, model):
    global win_counter, lose_counter
    temp = maze

    env.changeMap(temp)

    env.countWin(0)
    env.countLose(0)
    win_counter = 0
    lose_counter = 0
    play_game(model, Qmaze_.Qmaze(temp), (0, 0))  # 0의 위치에서 게임 시작
    env.reset()  # 리셋


if __name__ == "__main__":
    print("0: 새로 학습 1: 기존 것에 학습 2: 학습 결과 보기")
    a = input()

    total_map = []
    f = open('D:/GoogleDrive/졸프/원본/maze_주석/트레이닝맵.txt', 'r+')

    while True:
        x = np.zeros((6, 4))  # 세로, 가로
        i = 0
        while True:
            line = f.readline()  # 한 줄씩 읽어서
            if line == '.':
                total_map.append(x)
                break
            elif line == ',\n':
                i = 0
                total_map.append(x)
                break
            list_str = line.split()
            y = np.array(list_str, dtype='int')
            for c in range(4):
                x[i][c] = y[c] # 복사
            i += 1
        if line == '.':
            break
    f.close()

    # Load the model from disk
    model = qtrain_.build_model(maze)

    if (a == '0'):
        for i in range(len(total_map)):
            trainMat_new(total_map[i], env, model)
        trainMat_new(total_map[i], env, model)
        # load model
        model = qtrain_.build_model(maze)
        model.load_weights('model.h5')

    elif (a == '1'):
        for j in range(5):  # 반복횟수
            for i in range(len(total_map)):
                trainMat(total_map[i], env, model)
        # trainMat(total_map[i], env, model)

    elif (a == '2'):
        model.load_weights('model.h5') # 트레인 데이터를 불러옴

        while True:
            for i in range(len(total_map)):
                if not confirmResult(total_map[i], env, model):
                    trainMat(total_map[i], env, model)

                time.sleep(0.5)

