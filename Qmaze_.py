# -*- coding: utf-8 -*-
import numpy as np
import maze_
import start_this
import qtrain_

visited_mark = 0.8  # Cells visited by the rat will be painted by gray 0.8
rat_mark = 4  # The current rat cell will be painteg by gray 0.5

LEFT = 0
UP = 1
RIGHT = 2
DOWN = 3

env = maze_.env

increment_reward = 0

class Qmaze(object):
    def __init__(self, maze, rat=(0, 0)):  # maze는 맵 그린 array, rat은 로봇 좌표


        self._maze = np.array(maze)  # array 복사
        nrows, ncols = self._maze.shape  # 맵의 가로 세로 길이

        # print("타겟: ",self.target[0][0], self.target[1][0])
        m, n = np.where(maze == 3)  # array에서 2를 찾고 행 렬 성분을 가짐 pks
        self.target = (m, n)  # 목표점

        # 1인 부분은 free_cell이라 지정 ( 이중 for문 )
        # free_cells는 (r,c) free_cell 좌표를 담은 배열
        self.block_cells = [(r, c) for r in range(nrows) for c in range(ncols) if self._maze[r, c] == 0.0]
        self.free_cells = [(r, c) for r in range(nrows) for c in range(ncols) if self._maze[r, c] == 1.0]
        self.waypoint = [(r, c) for r in range(nrows) for c in range(ncols) if self._maze[r, c] == 2.0]  # 경유지

        self.total_waypoint_size = len(self.waypoint)
        self.waypoint_count = 0
        # 목표점은 free_cell이 아니기 때문에 배열에서 제거
        #self.free_cells.remove(self.target)

        # 목표점을 0인 곳에 정의 했을 때
        if self._maze[self.target] == 0.0:
            raise Exception("Invalid maze: target cell cannot be blocked!")

        # 로봇의 위치를 범위 밖에 정의했을 때
        if not rat in self.free_cells:
            raise Exception("Invalid Rat Location: must sit on a free cell")
        self.reset(rat)

    # 리셋하는 부분
    def reset(self, rat):
        self.rat = rat  # 로봇 좌표

        self.maze = np.copy(self._maze)  # 맵 배열을 다시 받아옴

        row, col = rat
        #self.maze[row, col] = rat_mark  # 로봇 위치 표시해 줌
        self.state = (row, col, 'start')  # 스테이트 초기화 - > start
        self.min_reward = -0.5 * self.maze.size  # size 값은 가로 곱하기 세로
        self.total_reward = 0  # total reward 초기화
        self.waypoint_count = 0 # 방문 횟수 초기화
        self.visited = set()  # 방문 배열 초기화
        self.visited_waypoint = set()  # 경유지 방문 배열 초기화

    # state 업데이트
    def update_state(self, action):

        # 로봇가로좌표, 로봇세로좌표, 상태를 받아옴 (스테이트 정보)
        nrow, ncol, nmode = rat_row, rat_col, mode = self.state
        # print ("스테이트 좌표", self.state)
        # print(self._maze)

        if self.maze[rat_row, rat_col] == 2.0:  # 현재 있는 곳이 경유지이면
            self._maze[rat_row, rat_col] = 1.0
            self.free_cells.append((rat_row, rat_col))
            self.visited.add((rat_row, rat_col))

            # self.visited_waypoint.add((rat_row, rat_col))  # 경유지 좌표담은 배열

        elif 0.0 < self.maze[rat_row, rat_col] < 2.0:  # 현재 있는 곳이 벽이 아니면
            self.visited.add((rat_row, rat_col))  # 방문한 좌표담은 배열

        # print(self.visited)

        valid_actions = self.valid_actions()  # 현재 스테이트에서 갈수 있는 방향(액션) 만 담은 배열

        if not valid_actions:  # 갈 곳이 없을 때 스테이트 정보에 blocked를 넣음
            nmode = 'blocked'

        elif action in valid_actions:  # 갈 곳이 있을 때
            nmode = 'valid'

            if action == LEFT:
                ncol -= 1  # 왼쪽으로 한칸 이동
            elif action == UP:
                nrow -= 1  # 위쪽으로 한칸 이동
            elif action == RIGHT: # if
                ncol += 1  # 오른쪽으로 한칸 이동
            elif action == DOWN:
                nrow += 1  # 아래쪽으로 한칸 이동
            else:  # invalid action, no change in rat position
                nmode = 'invalid'

        self.state = (nrow, ncol, nmode)  # 새로 정의한 스테이트 값 대입

    # reward 받는 부분
    def get_reward(self):
        rat_row, rat_col, mode = self.state  # 스테이트 정보를 받아옴
        nrows, ncols = self.maze.shape  # 맵 가로 세로 사이즈

        global increment_reward
        increment_reward = 10

        if rat_row == self.target[0] and rat_col == self.target[1]:  # 목적지 도착 시 리워드
            if self.waypoint_count == self.total_waypoint_size:
                increment_reward += 50
                return increment_reward # 6
            else:
                # return 0
                # print("다 안밟음")
                return -20  # lose 조건에 만족

        if (rat_row, rat_col) in self.visited:  # 방문한 곳은 -0.25 리워드
            # print ("재방문")
            return -10

        if mode == 'valid':  # 유효한 곳은 -0.04 리워드
            if (rat_row, rat_col) in self.waypoint:  # 경유지 도착 시 리워드 #pks
                # print("경유지", self.waypoint)
                self.waypoint.remove((rat_row, rat_col))
                # print("경유지삭제후", self.waypoint)
                env.waypoint_color_change(rat_row, rat_col)
                self.waypoint_count += 1
                increment_reward += 5

                self.maze[rat_row, rat_col] = 1.0

                return increment_reward  # 5
            elif (rat_row, rat_col) in self.block_cells:  # 경유지 도착 시 리워드 #pks
                print("벽 밟ㄷ았다")
                return -20
            else:
                return -0.4  # - 0.04

    def act(self, action):
        # print("액션: ", action)
        # print("변경 전 스테이트: ", self.state, " ", self.total_reward)
        self.update_state(action)  # 액션을 수행하고 state를 업데이트함
        # print("변경 후 스테이트: ", self.state, " ", self.total_reward)
        reward = self.get_reward()  # 새로운 스테이트에 대한 리워드를 받음
        self.total_reward += reward  # total_reward 에 방금 받은 리워드를 더함
        env.countUP(self.total_reward)

        status = self.game_status()  # win / lose / not_over
        envstate = self.observe()  # 새로 관측한 맵을 어레이를 한줄로 만듬 ( 로봇의 위치로 새로 반영됨)

        # print("total_reward: ", self.total_reward)
        return envstate, reward, status  # 관측 결과, 보상, 상태

    def observe(self):
        canvas = self.draw_env()  # 로봇의 위치를 새로 적은 어레이
        envstate = canvas.reshape((1, -1))  # 어레이를 한줄로 만듬
        #print("envstate: ", envstate)

        return envstate

    def draw_env(self):
        canvas = np.copy(self.maze)  # 맵 어레이 복사
        # print("canvas: " , canvas)
        nrows, ncols = self.maze.shape  # 맵 사이즈

        # 로봇의 위치를 0.5에서 다시 1로 바꿈.  다시 그릴거라서
        # for r in range(nrows):
        #     for c in range(ncols):
        #         if canvas[r, c] > 0.0:
        #             canvas[r, c] = 1.0

        # draw the rat
        row, col, valid = self.state  # 스테이트를 다시 받음
        canvas[row, col] = rat_mark  # 로봇 이동한 그림을 어레이로 다시 그림, 로봇 있는 곳은 0.5 벽은 0 길은 1

        return canvas

    def game_status(self):
        if self.total_reward < self.min_reward:  # 갈 곳이 없을 때 lose 출력
            return 'lose'

        rat_row, rat_col, mode = self.state  # 현재 스테이트를 확인함
        nrows, ncols = self.maze.shape  # 맵 배열 사이즈 얻음

        if rat_row == self.target[0] and rat_col == self.target[1]:  # 목적지 좌표랑 일치할 때 win

            if self.waypoint_count == self.total_waypoint_size:
                # print("total_reward: ", self.total_reward)
                # print("경유지 카운터 = ", self.waypoint_count)
                # print("경유지 토탈 = ", self.total_waypoint_size)
                return 'win'
            else:
                return 'lose'

        return 'not_over'  # 게임이 끝나지 않았을 때 출력

    # 가능한 액션들만 보여줌
    def valid_actions(self, cell=None):
        if cell is None:  # cell이 입력 안됐을 때
            row, col, mode = self.state  # 스테이트를 사용해서 현재 스테이트에서의 좌표를 얻어냄
        else:
            row, col = cell  # 입력 받았을 때 좌표

        actions = [0, 1, 2, 3]  # 액션 전체

        nrows, ncols = self.maze.shape  # 맵 사이즈

        if row == 0:  # 맨 위에 있을 때
            actions.remove(1)  # up 삭제
        elif row == nrows - 1:  # 맨 아래에 있을 때
            actions.remove(3)  # down 삭제

        if col == 0:  # 맨 왼쪽에 있을 때
            actions.remove(0)  # left 삭제
        elif col == ncols - 1:  # 맨 오른쪽에 있을 때
            actions.remove(2)  # right 삭제
        return actions  # 가능한 액션 배열

    # 가능한 액션들만 보여줌
    def valid_actions2(self, cell=None):
        if cell is None:  # cell이 입력 안됐을 때
            row, col, mode = self.state  # 스테이트를 사용해서 현재 스테이트에서의 좌표를 얻어냄
        else:
            row, col = cell  # 입력 받았을 때 좌표

        actions = [0, 1, 2, 3]  # 액션 전체

        nrows, ncols = self.maze.shape  # 맵 사이즈

        if row == 0:  # 맨 위에 있을 때
            actions.remove(1)  # up 삭제
        elif row == nrows - 1:  # 맨 아래에 있을 때
            actions.remove(3)  # down 삭제

        if col == 0:  # 맨 왼쪽에 있을 때
            actions.remove(0)  # left 삭제
        elif col == ncols - 1:  # 맨 오른쪽에 있을 때
            actions.remove(2)  # right 삭제

        if row > 0 and self.maze[row - 1, col] == 0.0:  # 맨 위쪽이 아니고 윗 칸이 막혀있을 때
            actions.remove(1)  # up 삭제

        if row < nrows - 1 and self.maze[row + 1, col] == 0.0:  # 맨 아래쪽이 아니고 아래 칸이 막혀있을 때
            actions.remove(3)  # down 삭제

        if col > 0 and self.maze[row, col - 1] == 0.0:  # 맨 왼쪽이 아니고 왼쪽 칸이 막혀있을 때
            actions.remove(0)  # left 삭제

        if col < ncols - 1 and self.maze[row, col + 1] == 0.0:  # 맨 오른쪽이 아니고 오른쪽 칸이 막혀있을 때
            actions.remove(2)  # right 삭제

        return actions  # 가능한 액션 배열
