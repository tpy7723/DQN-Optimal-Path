# -*- coding: utf-8 -*-

import cv2 as cv
import numpy as np
import PIL.Image as pilimg
import matplotlib.pyplot as plt
import math

#image.shape[0] : 세로길이
#image.shape[1] : 가로길이
#image.shape[2] : 배열 안의 인자 개수
#HSV : 0-red, 60-yellow, 120-green, 180-하늘색, 240-blue, 300-pink, 360-red


image = cv.imread("img0514.png", cv.IMREAD_COLOR)
# print (image.shape)
image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
image = np.array(image)
#print(image[201])
#print(image.size/4)
#print(image.shape[0])

sum_R = 0
sum_G = 0
sum_B = 0
#print("SSS", image[201][1][2])
i=0
j=0


myMaze = np.zeros((6,4))

for col in range(6):

    for row in range(4):
        sum_R=0
        sum_G=0
        sum_B=0
        avg_R=0
        avg_G=0
        avg_B=0

        for y in range(col*math.floor(image.shape[0]/6), (col+1)*math.floor(image.shape[0]/6)):

            for x in range(row*math.floor(image.shape[1]/4), (row+1)*math.floor(image.shape[1]/4)):
                if y < image.shape[0]:
                    if x < image.shape[1]:
                        sum_R += image[y][x][0]
                        sum_G += image[y][x][1]
                        sum_B += image[y][x][2]

        avg_R = sum_R / (math.floor(image.shape[0] / 6) * math.floor(image.shape[1] / 4))
        avg_G = sum_G / (math.floor(image.shape[0] / 6) * math.floor(image.shape[1] / 4))
        avg_B = sum_B / (math.floor(image.shape[0] / 6) * math.floor(image.shape[1] / 4))
#        print("[", col, "][", row, "]", "R:", avg_R, "G:", avg_G, "B:", avg_B)

        if avg_R < 100 and avg_G < 100 and avg_B > 200:
            myMaze[col][row] = 2
        elif avg_R < 100 and avg_G < 100 and avg_B < 100:
            myMaze[col][row] = 0
        elif avg_R > 200 and avg_G > 200 and avg_B > 200:
            myMaze[col][row] = 1
        elif avg_R > 200 and avg_G > 200:
            myMaze[col][row] = 3

        i += math.floor(image.shape[1]/4)
    j += math.floor(image.shape[0]/6)

#
# print(myMaze)
#
# plt.imshow(image)
# plt.show()