# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import pickle

import numpy as np
import Qmaze_
import maze_
import qtrain_
import time, datetime
import image_

maze = maze_.total_maze
env = maze_.env

def play_game(model, qmaze, rat_cell):
    global win_counter, lose_counter
    # env.changeMap(qmaze._maze)
    qmaze.reset(rat_cell)  # 입력 받은 위치로 로봇을 맵에 초기화함
    envstate = qmaze.observe()  # 현재 스테이트를 받아옴

    mytime = 0.3
    while True:

        env.render()
        time.sleep(mytime)

        prev_envstate = envstate  # 스테이트를 전환
        # print("envstate: " , envstate)
        # get next action
        q = model.predict(prev_envstate)  # 스테이트를 넣고 q 배열이 나옴
        # print("q: ", q)

        action = np.argmax(q[0])  # 큰 Q 벨류가 액션

        valid_actions2 = qmaze.valid_actions2()
        arg_n = 2
        while action not in valid_actions2:
            #print(model.predict(prev_envstate).argsort())
            action = model.predict(prev_envstate).argsort()[0][arg_n]
            arg_n = arg_n - 1

        # apply action, get rewards and new state
        env.step(action)  # 액션 취함
        # env.countUP(self.total_reward)


        envstate, reward, game_status = qmaze.act(action)

        if game_status == 'win':  # 목적지에 도착하면
            print("win\n")  # win
            win_counter += 1
            env.countWin(win_counter)
            time.sleep(mytime)
            return True  # true

        elif game_status == 'lose':  # 갈 곳이 없으면
            print("lose\n")  # lose
            lose_counter += 1
            env.countLose(lose_counter)
            time.sleep(mytime)
            return False  # false


def trainMat(maze, env, model):
    global win_counter, lose_counter
    temp = maze

    # env.changeMap(temp) # 맵 초기화: 로봇, 경유지, 벽

    env.countWin(0)
    env.countLose(0)
    win_counter = 0
    lose_counter = 0

    # 기존 것에 더 학습
    qtrain_.qtrain(model, temp, epochs=1000, max_memory=80 * maze.size, data_size=32, weights_file="model.h5")
    # model.save('model.h5')



def trainMat_new(maze, env, model):
    global win_counter, lose_counter
    temp = maze

    # env.changeMap(temp)
    env.countWin(0)
    env.countLose(0)
    win_counter = 0
    lose_counter = 0
    # 기존 것에 더 학습
    qtrain_.qtrain(model, temp, epochs=700, max_memory=80* maze.size, data_size=32)



def confirmResult(maze, env, model):
    global win_counter, lose_counter
    temp = maze
    qmaze = Qmaze_.Qmaze(temp)
    # print(temp)
    env.changeMap(temp)

    env.countWin(0)
    env.countLose(0)
    win_counter = 0
    lose_counter = 0

    if play_game(model, qmaze, qmaze.initialrat):  # 0의 위치에서 게임 시작
        return True
    else:
        return False

if __name__ == "__main__":
    print("0: 새로 학습 1: 기존 것에 학습 2: 학습 결과 보기 3: 특정 맵 학습 하기 4: 특정 맵 결과 보기")
    a = input()

    total_map = []
    f = open('트레이닝맵.txt', 'r+')

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
                x[i][c] = y[c]  # 복사
            i += 1
        if line == '.':
            break
    f.close()

    print("트레이닝 맵 총 개수: " , len(total_map) )
    start_time = datetime.datetime.now()  # 시작 시간

    # Load the model from disk
    model = qtrain_.build_model(maze)
    if (a == '0'):
        for j in range(1):  # 반복횟수
            for i in range(len(total_map)):
                env.countRepeat(j + 1, i + 1)
                trainMat_new(total_map[i], env, model)
            qtrain_.experience.save()

    elif (a == '1'):
        # trainMat(total_map[6], env, model)
        # trainMat(total_map[5], env, model)
        for j in range(1):  # 반복횟수
            for i in range(len(total_map)):
                env.countRepeat(j+1, i+1)
                trainMat(total_map[i], env, model)
                print("메모리 길이: ",len(qtrain_.experience.memory))
        qtrain_.experience.save()
                # trainMat(total_map[i], env, model)

        # time.sleep(10)
        # qtrain_.my_train()

    elif (a == '2'):
        model.load_weights('model.h5')  # 트레인 데이터를 불러옴
        win_map = []
        fail_map=[]

        win = 0
        lose = 0
        while True:
            for i in range(len(total_map)):
                env.countUP(0)

                env.countRepeat(1,i+1)
                if confirmResult(total_map[i], env, model): # image_.myMaze
                    # win_map.append(total_map[i])
                    win += 1
                else:
                    # fail_map.append(total_map[i])
                    lose += 1
                time.sleep(1) # 1로 바꿔줌
                    # trainMat(total_map[i], env, model)
                    # model.load_weights('model.h5')  # 트레인 데이터를 불러옴
                # time.sleep(0.5)

        # with open('win_map.p','wb') as file:
        #     pickle.dump(win_map, file)
        #
        # with open('fail_map.p','wb') as file:
        #     pickle.dump(fail_map, file)
        # print("성공 갯수: %d, 실패 갯수: %d" % (win, lose))

    elif (a == '3'):
        # 0은 까만 벽 1은 하얀 벽 2는 경유지 3은 목적지 4는 로봇
        # total_maze = np.array([
        #     [1., 1., 1., 1.],
        #     [0., 0., 2., 2.],
        #     [2., 2., 2., 1.],
        #     [1., 1., 0., 0.],
        #     [2., 2., 1., 1.],
        #     [1., 1., 2., 3.]
        # ])

        trainMat(image_.myMaze, env, model)
        qtrain_.experience.save()

    elif (a == '4'):
        model.load_weights('model.h5')  # 트레인 데이터를 불러옴

        # 0은 까만 벽 1은 하얀 벽 2는 경유지 3은 목적지 4는 로봇
        # total_maze = np.array([
        #     [1., 1., 1., 1.],
        #     [0., 0., 2., 2.],
        #     [2., 2., 2., 1.],
        #     [1., 1., 0., 0.],
        #     [2., 2., 1., 1.],
        #     [1., 1., 2., 3.]
        # ])

        total_maze = np.array([
            [4., 1., 1., 1.],
            [0., 0., 2., 1.],
            [1., 1., 1., 1.],
            [1., 1., 0., 0.],
            [1., 1., 1., 1.],
            [1., 1., 1., 1.]
        ])
        print(type(total_maze))
        while True:
            confirmResult(total_maze, env, model)
    elif (a == '5'):
        qtrain_.my_train()

    elif (a == '6'):
        with open('fail_map.p','rb') as file:
            try:
                fail_map = pickle.load(file)
            except EOFError:
                print("EOF error 발생")
        for j in range(1):  # 반복횟수
            for i in range(len(fail_map)):
                env.countRepeat(j+1, i+1)
                trainMat(fail_map[i], env, model)
                print("메모리 길이: ",len(qtrain_.experience.memory))
        # qtrain_.experience.save()
    elif (a == '7'):
        with open('win_map.p','rb') as file:
            try:
                win_map = pickle.load(file)
            except EOFError:
                print("EOF error 발생")
        for j in range(1):  # 반복횟수
            for i in range(len(win_map)):
                env.countRepeat(j+1, i+1)
                trainMat(win_map[i], env, model)
                print("메모리 길이: ",len(qtrain_.experience.memory))
    elif (a == '8'):
        model.load_weights('model.h5')  # 트레인 데이터를 불러옴
        while True:
            # image_.takeapicture()
            temp = image_.takeapicture()
            print(type(temp))
            confirmResult(temp, env, model)

    end_time = datetime.datetime.now()
    dt = end_time - start_time  # 시간 차이
    seconds = dt.total_seconds()  # 시간 차이를 초로 바꿈

    t = qtrain_.format_time(seconds)  # 시간 형식

    print("총 걸린시간: %s" % t)