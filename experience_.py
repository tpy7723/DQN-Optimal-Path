# -*- coding: utf-8 -*-
import numpy as np
import pickle
from random import *

# 경험 기록
class Experience(object):
    # model:  neural network model ,
    # max_memory : maximal length of episodes to keep. When we reach the maximal length of memory,
    # each time we add a new episode, the oldest episode is deleted
    # discount : discount factor 최단경로에 도움됨

    def __init__(self, model, max_memory=100, discount=0.95): # 생성자
        self.model = model
        self.max_memory = max_memory
        self.discount = discount
        self.memory = list() # 에피소드를 보관할 리스트
        self.num_actions = model.output_shape[-1]

    def save(self):
        with open('experience_memory.p','wb') as file:
            pickle.dump(self.memory, file)

    def load(self):
        with open('experience_memory.p','rb') as file:
            self.memory = pickle.load(file)
            # print(self.memory)

    # 메모리에 에피소드 추가하는 함수
    def remember(self, episode):
        self.memory.append(episode) # 에피소드에 추가

        if len(self.memory) > self.max_memory: # 메모리가 최대 메모리보다 클 경우
            del self.memory[randint(0,len(self.memory)-1)] # 리스트의 첫번 째 칸을 지움  the oldest episode is deleted

    # 예측
    def predict(self, envstate): # 로봇의 위치와 맵을 한줄로 나타낸 배열을 인풋으로 사용
        return self.model.predict(envstate)[0]


    def countLength(self):  # 로봇의 위치와 맵을 한줄로 나타낸 배열을 인풋으로 사용
        return self.model

    def get_data(self, data_size=10):
        env_size = self.memory[0][0].shape[1]  # envstate 1d size (1st element of episode) 한줄로 된 맵 어레이 원소 개수
        mem_size = len(self.memory) # 메모리 사이즈
        data_size = min(mem_size, data_size) # 최대 사이즈는 data_size가 될 것

        inputs = np.zeros((data_size, env_size))
        # print(inputs)
        targets = np.zeros((data_size, self.num_actions))


        for i, j in enumerate(np.random.choice(range(mem_size), data_size, replace=False)):
            envstate, action, reward, envstate_next, game_over = self.memory[j]  # 메모리에서 에피소드를 꺼냄
            inputs[i] = envstate # 데이터 사이즈 만큼 상태들을 저장함

            # There should be no target values for actions not taken.
            targets[i] = self.predict(envstate) # 현재 스테이트의 벨류
            Q_sa = np.max(self.predict(envstate_next)) # 다음 스테이트에서의 최대 벨류
            if game_over: # 게임이 끝나면
                targets[i, action] = reward # 현재 리워드를 넣음
            else: # 게임이 끝나지 않았으면
                # reward + gamma * max_a' Q(s', a')
                targets[i, action] = reward + self.discount * Q_sa # 예측 값을 넣음

        return inputs, targets # 스테이트 저장소, Q 테이블