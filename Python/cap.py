# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 19:10:01 2020

@author: 爬上阁楼的鱼
"""


import cv2
import sys

# 根据摄像头设置IP及rtsp端口
url = 'rtsp://admin:hk123456@192.168.1.52:554/h264/ch42/sub/av_stream'
#  33  34  35  36  37  38  39  40x 41  42  43  44  45  46  47  48  49x

print('Start!!!')

# 读取视频流
cap = cv2.VideoCapture(url)

print('Start OK!!!')

# 设置视频参数
# cap.set(3, 480)

print(cap.isOpened())

print(sys.version)

print(cv2.__version__)

while cap.isOpened():
    ret_flag, img_camera = cap.read()
    cv2.imshow("camera", img_camera)

    # 每帧数据延时 1ms, 延时为0, 读取的是静态帧
    k = cv2.waitKey(1)
    if k == ord('s'):
        cv2.imwrite("test.jpg", img_camera)
    if k == ord('1'):
        break

# 释放所有摄像头
cap.release()

# 删除窗口
cv2.destroyAllWindows()
