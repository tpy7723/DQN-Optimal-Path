# -*- coding: utf-8 -*-\
import numpy as np
import keras.backend.tensorflow_backend as KK
from keras.layers.core import Dense
from keras.models import Sequential, load_model
from keras.layers.advanced_activations import PReLU
import datetime, random, json
import maze_
import Qmaze_
import experience_
import start_this
import matplotlib.pyplot as plt
import time

# env = start_this.env
env = maze_.env

LEFT = 0
UP = 1
RIGHT = 2
DOWN = 3

# Actions dictionary
actions_dict = {
    LEFT: 'left',
    UP: 'up',
    RIGHT: 'right',
    DOWN: 'down',
}

# Exploration factor 탐험률 0.1
epsilon = 0.15

# 액션의 수 (상하 좌우)  = 4
num_actions = len(actions_dict)

y = []
y_ = []

# 맵에 대해 모델 생성하는 함수
def build_model(maze):
    model = Sequential()  # 모델 생성 ann
    model.add(Dense(maze.size *3, input_shape=(maze.size,)))  # input , 1st hidden
    model.add(PReLU())  # activation function
    model.add(Dense(maze.size*2))  # 2nd hidden
    model.add(PReLU())  # activation function
    model.add(Dense(maze.size * 1))  # 2nd hidden
    model.add(PReLU())  # activation function
    model.add(Dense(num_actions))  # output
    model.compile(optimizer='adam', loss='mse')

    return model

# Load the model from disk
temp_model = build_model(maze_.total_maze)

# # Initialize experience replay object
# experience = experience_.Experience(temp_model, max_memory=max_memory)

experience = experience_.Experience(temp_model, max_memory=1920)
experience_exist = 0

def qtrain(model, maze, **opt):
    global epsilon, max_epoch, temp_epsilon, experience, experience_exist
    start_this.env.reset()
    # print("답",maze)
    # env = maze_.Maze(maze)
    # env.reset()



    epsilon = 0.15
    counter = 0
    n_epoch = opt.get('epochs', 15000)  # 15000 epoch 횟수
    max_memory = opt.get('max_memory', 1000)  # 128

    data_size = opt.get('data_size', 50)  # 32

    weights_file = opt.get('weights_file', "")
    name = opt.get('name', 'model')  # model

    start_time = datetime.datetime.now()  # 시작 시간

    # If you want to continue training from a previous model,
    # just supply the h5 file name to weights_file option

    if weights_file:  # weights_file을 입력했을 때
        print("loading weights from file: %s" % (weights_file,))
        model.load_weights(weights_file)  # weight_file을 로드 함

    # Construct environment/game from numpy array: maze (see above)

    qmaze = Qmaze_.Qmaze(maze)

    # experience = experience_.Experience(model, max_memory=max_memory)
    # # Initialize experience replay object
    if experience_exist == 0:
        experience = experience_.Experience(model, max_memory=192000)
        experience.load()
        experience_exist = 1

    win_history = []  # history of win/lose game
    # hsize = qmaze.maze.size // 2  # history window size = 8
    hsize = 8
    win_rate = 0.0  # 성공률 초기화

    for epoch in range(n_epoch):  # epoch 수 만큼 반복
        max_epoch = epoch
        total_reward = 0

        loss = 0.0  # loss 0으로 초기화
        rat_cell = random.choice(qmaze.free_cells)  # 랜덤위치에서 시작
        qmaze.reset((0, 0))  # start 스테이트로 바꾸고 랜덤위치에서 로봇 생성 후 배열에 적용
        game_over = False  # game_over X

        # get initial envstate (1d flattened canvas)

        qmaze = Qmaze_.Qmaze(maze)
        envstate = qmaze.observe()  # 초기 상태 어레이를 한 줄로 표현

        n_episodes = 0  # 에피소드 초기화

        start_this.env.reset_location((0, 0))  # rat cell

        while not game_over:  # 게임이 끝날 때 까지

            valid_actions = qmaze.valid_actions()  # 가능한 액션을 얻는다
            valid_actions2 = qmaze.valid_actions2()  # 가능한 액션을 얻는다

            if not valid_actions: break  # 가능한 액션이 없으면 break
            prev_envstate = envstate  # 이전 스테이트에 현재 스테이트를 넣음
            # print("가능한 액션 ", valid_actions)
            # print(envstate)
            # print(experience.predict(prev_envstate))
            # Get next action
            if np.random.rand() < epsilon:  # 입실론 보다 작으면
                action = random.choice(valid_actions2)  # 탐험  여기수정해야겟다***********
                # print(valid_actions2)
            else:  # 입실론 보다 크면
                action = np.argmax(experience.predict(prev_envstate))  # 최대 Q 에 맞춰서 행동
                # print(experience.predict(prev_envstate))
                # print(valid_actions2)
                # arg_n = 2
                # while action not in valid_actions2:
                #     action = experience.predict(prev_envstate).argsort()[arg_n]
                #     arg_n = arg_n - 1

            # Apply action, get reward and new envstate
            envstate, reward, game_status = qmaze.act(action)  # 액션을 취하고 리워드와 새 스테이트를 받는다
            total_reward += reward
            # time.sleep(3)
            start_this.env.step(action)

            if game_status == 'win':  # 목적지에 도착했을 때
                win_history.append(1)  # 히스토리에 1 추가
                game_over = True  # 게임 끝
                start_this.env.reset()  # 로봇을 초기 위치로 감

            elif game_status == 'lose':  # 게임에 졌을 때
                win_history.append(0)  # 히스토리에 0 추가
                game_over = True  # 게임 끝
                start_this.env.reset()  # 로봇을 초기 위치로 감
            else:
                game_over = False  # 게임 안끝남

            # Store episode (experience)
            episode = [prev_envstate, action, reward, envstate, game_over]  # 이전스테이트, 액션, 보상, 현재스테이트, 게임끝 여부   =>  에피소드
            experience.remember(episode)  # 메모리에 에피소드 추가하는 함수
            n_episodes += 1  # 에피소드 카운트

            # Train neural network model
            # print("학습")
            inputs, targets = experience.get_data(data_size=8)  # 타겟은 예측값
            h = model.fit(
                inputs,
                targets,
                epochs=4,  # 학습 데이터 전체셋을 몇 번 학습하는지를 의미합니다. 동일한 학습 데이터라고 하더라도 여러 번 학습할 수록 학습 효과는 커집니다.
                # 하지만, 너무 많이 했을 경우 모델의 가중치가 학습 데이터에 지나치게 최적화되는 과적합(Overfitting) 현상이 발생합니다.
                batch_size=8,  # 만약 batch_size가 10이라면, 총 10개의 데이터를 학습한 다음 가중치를 1번 갱신하게 됩니다.
                # batch_size 값이 크면 클수록 여러 데이터를 기억하고 있어야 하기에 메모리가 커야 합니다. 그대신 학습 시간이 빨라집니다.
                # batch_size 값이 작으면 학습은 꼼꼼하게 이루어질 수 있지만 학습 시간이 많이 걸립니다.
                verbose=0,  # Integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
            )

            # X : 입력 데이터
            # Y : 결과(Label 값) 데이터
            # epochs : 학습 데이터 반복 횟수
            # batch_size : 한 번에 학습할 때 사용하는 데이터 개수

            loss = model.evaluate(inputs, targets, verbose=0)

        if len(win_history) > hsize:
            win_rate = sum(win_history[-hsize:]) / hsize  # 최근 8개 데이터

        dt = datetime.datetime.now() - start_time
        t = format_time(dt.total_seconds())  # 시간 계산

        template = "Epoch: {:03d}/{:d} | Loss: {:.4f} | Episodes: {:d} | Win count: {:d} | Win rate: {:.3f} | time: {}"
        if total_reward > 5 and counter == 0:
            # raiseup = epoch
            counter += 1
        y_.append(total_reward)
        y.append(loss)
        print(template.format(epoch, n_epoch - 1, loss, n_episodes, sum(win_history), win_rate, t))

        env.countWinrate(win_rate)
        env.countWin(sum(win_history))
        env.countLose(epoch - sum(win_history))
        # we simply check if training has exhausted all free cells and if in all

        # cases the agent won

        if win_rate >= 0.875:
            temp_epsilon = 0.005  # 성공률 90프로 일 때 입실론 값 대폭 감소
        elif win_rate >= 0.750:
            temp_epsilon = 0.01  # 성공률 90프로 일 때 입실론 값 대폭 감소
        elif win_rate >= 0.625:
            temp_epsilon = 0.015  # 성공률 90프로 일 때 입실론 값 대폭 감소
        elif win_rate >= 0.500:
            temp_epsilon = 0.02  # 성공률 90프로 일 때 입실론 값 대폭 감소
        elif win_rate >= 0.375:
            temp_epsilon = 0.04  # 성공률 90프로 일 때 입실론 값 대폭 감소
        elif win_rate >= 0.250:
            temp_epsilon = 0.06  # 성공률 90프로 일 때 입실론 값 대폭 감소
        elif win_rate >= 0.125:
            temp_epsilon = 0.1  # 성공률 90프로 일 때 입실론 값 대폭 감소
        elif win_rate >= 0.000:
            temp_epsilon = 0.15  # 성공률 90프로 일 때 입실론 값 대폭 감소

        if epsilon > temp_epsilon:
            epsilon = temp_epsilon

        if sum(win_history[-hsize:]) == 0:
            epsilon = 0.1

        env.countEpsilon(epsilon)
        # print("epsilon = ", epsilon)
        # and completion_check(model, qmaze)

        if sum(win_history[-hsize:]) >= hsize and completion_check(model, maze):  # 모든 셀에 대해 검사
            print("Reached 100%% win rate at epoch: %d" % (epoch,))
            break

    # Save trained model weights and architecture, this will be used by the visualization code
    h5file = name + ".h5"
    json_file = name + ".json"
    model.save_weights(h5file, overwrite=True)

    with open(json_file, "w") as outfile:
        json.dump(model.to_json(), outfile)

    end_time = datetime.datetime.now()
    dt = end_time - start_time  # 시간 차이
    seconds = dt.total_seconds()  # 시간 차이를 초로 바꿈

    t = format_time(seconds)  # 시간 형식

    print('files: %s, %s' % (h5file, json_file))
    print("n_epoch: %d, max_mem: %d, data: %d, time: %s" % (epoch, max_memory, data_size, t))

    # x = range(0, max_epoch + 1)

    # fig = plt.figure()
    #
    # ax1 = fig.add_subplot(2, 1, 1)
    # ax2 = fig.add_subplot(2, 1, 2)
    # ax1.set_xlabel('Epoch')
    # ax1.set_ylabel('Loss')
    # ax2.set_xlabel('Epoch')
    # ax2.set_ylabel('Total Reward')
    # ax1.plot(x, y)
    # ax2.plot(x, y_)
    # ax1.axhline(y=3, color='r', linestyle='--', linewidth=2)
    # ax2.axvline(x=raiseup - 1, color='r', linestyle='--', linewidth=2)
    # ax1.grid()
    # ax2.grid()
    # plt.show(block=False)
    #
    # f1 = open("C:/Users/pks01/Desktop/졸프/x축.txt", 'w')
    # # f1.write("[")
    # for i in range(0, max_epoch + 1):
    #     data = str(x[i])
    #     f1.write(data)
    #     if i != max_epoch:
    #         f1.write(",")
    # # f1.write("]")
    #
    # f1.close()
    #
    # f2 = open("C:/Users/pks01/Desktop/졸프/y축.txt", 'w')
    # # f2.write("[")
    # for i in range(0, len(y)):
    #     data = str(y[i])
    #     f2.write(data)
    #     if i != len(y)-1:
    #         f2.write(",")
    # # f2.write("]")
    # f2.close()
    #
    # f3 = open("C:/Users/pks01/Desktop/졸프/y_축.txt", 'w')
    # # f3.write("[")
    # for i in range(0, len(y_)):
    #     data = str(y_[i])l
    #     f3.write(data)
    #     if i != len(y_)-1:
    #         f3.write(",")
    # # f3.write("]")
    # f3.close()

    return seconds


def completion_check(model, maze):
    # for cell in qmaze.free_cells:
    #
    #     if not qmaze.valid_actions(cell): # 가능한 길이 없으면 false
    #         return False
    #
    #     if not start_this.play_game(model, qmaze, cell): # 모든 셀에 대하여 갈 곳이 없으면
    #         return False
    cell = (0, 0)

    if start_this.confirmResult(maze, env, model):
        return True
    else:
        return False


# 시간 형식 지정
def format_time(seconds):
    if seconds < 400:
        s = float(seconds)
        return "%.1f seconds" % (s,)
    elif seconds < 4000:
        m = seconds / 60.0
        return "%.2f minutes" % (m,)
    else:
        h = seconds / 3600.0
        return "%.2f hours" % (h,)

def my_train():
    # Load the model from disk
    temp_model = build_model(maze_.total_maze)
    temp_model.load_weights("model.h5")

    experience = experience_.Experience(temp_model, max_memory=192000)
    experience.load()



    for i in range(1):
        inputs, targets = experience.get_data(data_size=192000)  # 타겟은 예측값

        h = temp_model.fit(
            inputs,
            targets,
            epochs=1,  # 학습 데이터 전체셋을 몇 번 학습하는지를 의미합니다. 동일한 학습 데이터라고 하더라도 여러 번 학습할 수록 학습 효과는 커집니다.
            # 하지만, 너무 많이 했을 경우 모델의 가중치가 학습 데이터에 지나치게 최적화되는 과적합(Overfitting) 현상이 발생합니다.
            batch_size=192000,  # 만약 batch_size가 10이라면, 총 10개의 데이터를 학습한 다음 가중치를 1번 갱신하게 됩니다.
            # batch_size 값이 크면 클수록 여러 데이터를 기억하고 있어야 하기에 메모리가 커야 합니다. 그대신 학습 시간이 빨라집니다.
            # batch_size 값이 작으면 학습은 꼼꼼하게 이루어질 수 있지만 학습 시간이 많이 걸립니다.
            verbose=0,  # Integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
        )
        loss = temp_model.evaluate(inputs, targets, verbose=0)
        print("loss: ", loss)

        # Save trained model weights and architecture, this will be used by the visualization code
        h5file = "model" + ".h5"
        json_file = "model" + ".json"
        temp_model.save_weights(h5file, overwrite=True)

        with open(json_file, "w") as outfile:
            json.dump(temp_model.to_json(), outfile)

    # X : 입력 데이터
    # Y : 결과(Label 값) 데이터
    # epochs : 학습 데이터 반복 횟수
    # batch_size : 한 번에 학습할 때 사용하는 데이터 개수
