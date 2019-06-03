# -*- coding: utf-8 -*-

import sys
import numpy as np
import Qmaze_
#import image_
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk


# total_maze = image_.myMaze

#0은 까만 벽 1은 하얀 벽 2는 경유지 3은 목적지 4는 로봇
total_maze = np.array([
    [1., 1., 1., 1.],
    [0., 0., 2., 2.],
    [2., 2., 2., 1.],
    [1., 1., 0., 0.],
    [2., 2., 1., 1.],
    [1., 1., 2., 3.]
])

m, n = np.where(total_maze == 3)  # array에서 2를 찾고 행 렬 성분을 가짐 pks

target_row = m
target_col = n  #목표점

UNIT = 40  # pixels 칸 사이즈

MAZE_H = total_maze.shape[0]  # grid height 창 높이
MAZE_W = total_maze.shape[1]  # grid width 창 너비

# 미로에 관한 클래스
class Maze(tk.Tk, object):
    def __init__(self, maze):
        super(Maze, self).__init__()
        self.robot_location = (0, 0)
        self.total_maze = maze
        self.label2 = tk.Label(self, text="Total Reward: " + "0" ,justify = "left")
        self.label2.pack()

        self.label3 = tk.Label(self, text="0") # 방향
        self.label3.pack()

        self.label4 = tk.Label(self, text="성공: " + "0")
        self.label4.pack()

        self.label5 = tk.Label(self, text="실패: " + "0")
        self.label5.pack()

        self.label6 = tk.Label(self, text="Win Rate: " + "0")
        self.label6.pack()

        self.label7 = tk.Label(self, text="epsilon: " + "0")
        self.label7.pack()

        self.label8 = tk.Label(self, text="회차/케이스: " + "0/0")
        self.label8.pack()

        self.title('maze')
        self.geometry('{0}x{1}+200+200'.format(self.total_maze.shape[0] * UNIT, self.total_maze.shape[1] * UNIT+300))
        self.build_maze()
        self.way_point = []




        # button = tk.Button(self, overrelief="solid", width=15, command=self.countUP, repeatdelay=1000,
        #                    repeatinterval=100)
        # button.pack()

    def countUP(self, total_reward):
        self.label2.config(text="Total Reward: " + str(total_reward))

    def countWin(self, win):
        self.label4.config(text="성공: " + str(win))

    def countLose(self, lose):
        self.label5.config(text="실패: " + str(lose))

    def countWinrate(self, winrate):
        self.label6.config(text="Win Rate: " + str(winrate))

    def countEpsilon(self, epsilon):
        self.label7.config(text="epsilon: " + str(epsilon))

    def countRepeat(self, repeat, case):
        self.label8.config(text="회차/케이스: " + str(repeat) + "/" + str(case))
    # def remove_waypoint(self):
    #     self.update()
    #     self.canvas.delete(self.way_point)

    def changeMap(self, maze):
        self.total_maze = maze
        self.update()
        self.canvas.delete("all")  # 맵에서 로봇을 지움
        i, j = np.where(self.total_maze == 0)  # array에서 0을 찾고 행 렬 성분을 가짐 pks
        k, l = np.where(self.total_maze == 2)  # array에서 2를 찾고 행 렬 성분을 가짐 pks
        m, n = np.where(self.total_maze == 3)  # array에서 2를 찾고 행 렬 성분을 가짐 pks

        # for i in range(0, k.size):
        #     self.way_point.append(0)

        # create origin
        origin = np.array([20, 20])  # 픽셀 크기 / 2 , 픽셀 크기 / 2

        for index in range(0, i.size):
            hell1_center = origin + np.array([UNIT * j[index], UNIT * i[index]])  # 열 / 행  #60 20
            # 검은 벽을 만듬
            self.hell = self.canvas.create_rectangle(
                hell1_center[0] - 20, hell1_center[1] - 20, hell1_center[0] + 20, hell1_center[1] + 20,  # 40,0,80,40,
                fill='black')

        # 목적지
        oval_center = origin + np.array([UNIT * n[0], UNIT * m[0]])
        self.oval = self.canvas.create_rectangle(
            oval_center[0] - 20, oval_center[1] - 20, oval_center[0] + 20, oval_center[1] + 20,
            fill='yellow')

        # 경유지
        for index in range(0, k.size):
            waypoint_center = origin + np.array([UNIT * l[index], UNIT * k[index]])  # 열 / 행  #60 20
            # 핑크 벽을 만듬
            self.way_point = self.canvas.create_rectangle(
                waypoint_center[0] - 20, waypoint_center[1] - 20, waypoint_center[0] + 20, waypoint_center[1] + 20,
                # 40,0,80,40,
                fill='blue')

        # 로봇
        robot_center = origin + np.array([UNIT * self.robot_location[1], self.robot_location[0]])  # 열 / 행  #60 20
        self.rect = self.canvas.create_rectangle(
            robot_center[0] - 20, robot_center[1] - 20, robot_center[0] + 20, robot_center[1] + 20,
            fill='red')
        self.canvas.pack()

    # 맵을 만듬
    def build_maze(self):
        self.canvas = tk.Canvas(self, bg='white',
                                height=self.total_maze.shape[0] * UNIT,
                                width=self.total_maze.shape[1] * UNIT)  # 표현할 창을 띄움

        i, j = np.where(self.total_maze == 0)  # array에서 0을 찾고 행 렬 성분을 가짐 pks
        k, l = np.where(self.total_maze == 2)  # array에서 2를 찾고 행 렬 성분을 가짐 pks
        m, n = np.where(self.total_maze == 3)  # array에서 2를 찾고 행 렬 성분을 가짐 pks

        # for i in range(0, k.size):
        #     self.way_point.append(0)

        # create origin
        origin = np.array([20, 20])  # 픽셀 크기 / 2 , 픽셀 크기 / 2

        for index in range(0, i.size):
            hell1_center = origin + np.array([UNIT * j[index], UNIT * i[index]])  # 열 / 행  #60 20
            # 검은 벽을 만듬
            self.hell = self.canvas.create_rectangle(
                hell1_center[0] - 20, hell1_center[1] - 20, hell1_center[0] + 20, hell1_center[1] + 20,  # 40,0,80,40,
                fill='black')

        # 목적지
        oval_center = origin + np.array([UNIT * n[0], UNIT * m[0]])
        self.oval = self.canvas.create_rectangle(
            oval_center[0] - 20, oval_center[1] - 20, oval_center[0] + 20, oval_center[1] + 20,
            fill='yellow')

        # 경유지
        for index in range(0, k.size):
            waypoint_center = origin + np.array([UNIT * l[index], UNIT * k[index]])  # 열 / 행  #60 20
            # 핑크 벽을 만듬
            self.way_point = self.canvas.create_rectangle(
                waypoint_center[0] - 20, waypoint_center[1] - 20, waypoint_center[0] + 20, waypoint_center[1] + 20,  # 40,0,80,40,
                fill='blue')

        # 로봇
        robot_center = origin + np.array([UNIT * self.robot_location[1], self.robot_location[0]])  # 열 / 행  #60 20
        self.rect = self.canvas.create_rectangle(
            robot_center[0] - 20, robot_center[1] - 20, robot_center[0] + 20, robot_center[1] + 20,
            fill='red')
        self.canvas.pack()

    # 경유지 밟았을 때
    def waypoint_color_change(self, raw, colomn):
        # create origin
        origin = np.array([20, 20])  # 픽셀 크기 / 2 , 픽셀 크기 / 2
        waypoint_center = origin + np.array([UNIT * colomn, UNIT * raw])  # 열 / 행  #60 20
        # 핑크 벽을 만듬
        self.white_point = self.canvas.create_rectangle(
            waypoint_center[0] - 20, waypoint_center[1] - 20, waypoint_center[0] + 20, waypoint_center[1] + 20,
            # 40,0,80,40,
            fill='white', outline='white')

        self.canvas.delete(self.rect)  # 맵에서 로봇을 지움
        robot_center = origin + np.array([UNIT * colomn, UNIT * raw])  # 열 / 행  #60 20
        self.rect = self.canvas.create_rectangle(
            robot_center[0] - 20, robot_center[1] - 20, robot_center[0] + 20, robot_center[1] + 20,
            fill='red')


    # 맵 초기화
    def reset(self):
        self.update()
        self.canvas.delete(self.rect)  # 맵에서 로봇을 지움

        robot_center = np.array([20, 20]) + np.array([UNIT * self.robot_location[1], UNIT * self.robot_location[0]])  # 시작지점 다시 만듬
        self.rect = self.canvas.create_rectangle(
            robot_center[0] - 20, robot_center[1] - 20, robot_center[0] + 20, robot_center[1] + 20,
            fill='red')

        # create origin
        origin = np.array([20, 20])  # 픽셀 크기 / 2 , 픽셀 크기 / 2

        k, l = np.where(self.total_maze == 2)  # array에서 2를 찾고 행 렬 성분을 가짐 pks

        # 경유지
        for index in range(0, k.size):
            waypoint_center = origin + np.array([UNIT * l[index], UNIT * k[index]])  # 열 / 행  #60 20
            # 핑크 벽을 만듬
            self.way_point = self.canvas.create_rectangle(
                waypoint_center[0] - 20, waypoint_center[1] - 20, waypoint_center[0] + 20, waypoint_center[1] + 20,
                # 40,0,80,40,
                fill='blue')

    # 다음 스테이트로 넘어가는 함수
    def step(self, action):
        s = self.canvas.coords(self.rect)  # 로봇 좌표를 받아옴

        base_action = np.array([0, 0])  # 1행 2열배열 , 배열 값은 0

        # 맵 안 벗어나게 하는 과정
        if action == 1:  # up
            if s[1] > UNIT:
                # print("up")
                self.label3.config(text="방향: " + "UP")
                base_action[1] -= UNIT
        elif action == 3:  # down
            if s[1] < (MAZE_H - 1) * UNIT:
                # print("down")
                self.label3.config(text="방향: " + "Down")
                base_action[1] += UNIT  # 상하는 두번쨰 값에 관여, up은 40뺴고 down은 40 더한다.
        elif action == 2:  # right
            if s[0] < (MAZE_W - 1) * UNIT:
                # print("right")
                self.label3.config(text="방향: " + "Right")
                base_action[0] += UNIT
        elif action == 0:  # left
            if s[0] >= UNIT:
                # print("left")
                self.label3.config(text="방향: " + "Left")
                base_action[0] -= UNIT  # 좌우는 두번쨰 값에 관여, left은 40뺴고 right은 40 더한다.

        # self.visitcolor = self.canvas.create_rectangle(
        #     s,
        #     fill='gray')

        self.canvas.move(self.rect, base_action[0], base_action[1])  # move agent  위에 것들은 무브하기 위해 구한 듯
        self.render()

    def render(self):
        self.update()

    def reset_location(self, robot):
        self.robot_location = robot
        self.canvas.delete(self.rect)  # 맵에서 로봇을 지움
        # 로봇
        robot_center = np.array([20, 20]) + np.array(
            [UNIT * self.robot_location[1], UNIT * self.robot_location[0]])  # 시작지점 다시 만듬
        self.rect = self.canvas.create_rectangle(
            robot_center[0] - 20, robot_center[1] - 20, robot_center[0] + 20, robot_center[1] + 20,
            fill='red')

env = Maze(total_maze)