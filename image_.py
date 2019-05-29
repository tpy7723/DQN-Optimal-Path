# -*- coding: utf-8 -*-

import cv2
import numpy as np
import PIL.Image as pilimg
import matplotlib.pyplot as plt
import math

#image.shape[0] : 세로길이
#image.shape[1] : 가로길이
#image.shape[2] : 배열 안의 인자 개수
#HSV : 0-red, 60-yellow, 120-green, 180-하늘색, 240-blue, 300-pink, 360-red

def Rotate(src):
    dst = cv2.transpose(src)
    dst = cv2.flip(dst, 1)
    return dst

cam = cv2.VideoCapture(1)
cam.set(3, 1280)  # CV_CAP_PROP_FRAME_WIDTH
cam.set(4, 720)  # CV_CAP_PROP_FRAME_HEIGHT
# cam.set(5,0) #CV_CAP_PROP_FPS
while True:
    ret_val, img = cam.read()  # 캠 이미지 불러오기
    img = Rotate(img)
    cv2.imshow("Cam Viewer", img)  # 불러온 이미지 출력하기
    if cv2.waitKey(1) == ord('q'):
        cv2.imwrite('img0514.jpg', img)
        break  # esc to quit

cam.release()
cv2.destroyAllWindows()

#image.shape[0] : 세로길이
#image.shape[1] : 가로길이
#image.shape[2] : 배열 안의 인자 개수
#HSV : 0-red, 60-yellow, 120-green, 180-하늘색, 240-blue, 300-pink, 360-red


image = cv2.imread("img0514.jpg", cv2.IMREAD_COLOR)
# print (image.shape)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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
        print("[", col, "][", row, "]", "R:", avg_R, "G:", avg_G, "B:", avg_B)

        if avg_R > 170 and avg_G < 170 and avg_B < 170:
            myMaze[col][row] = 2
        elif avg_R < 110 and avg_G < 110 and avg_B < 110:
            myMaze[col][row] = 0
        elif avg_R > 150 and avg_G > 150 and avg_B > 150:
            myMaze[col][row] = 1
        elif avg_B > 150 and avg_R < 130 and avg_G < 180:
            myMaze[col][row] = 3

        i += math.floor(image.shape[1]/4)
    j += math.floor(image.shape[0]/6)

#
print(myMaze)

myMaze2 = myMaze
#
# plt.imshow(image)
# plt.show()