# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 19:10:01 2020

@author: 爬上阁楼的鱼
"""

# 
import cv2
import sys

# 根据摄像头设置IP及rtsp端口
# url = 'rtsp://admin:Ilife2016@192.168.1.68:554/h264/ch38/main/av_stream'
# 33  34  35  36  37  38  39  40  41

# url = 'rtsp://admin:Ilife2016@192.168.1.7:554/h264/ch48/main/av_stream'
url = 'rtsp://admin:hk123456@192.168.1.52:554/h264/ch47/main/av_stream'
# #  33  34  35  36  37  38  39  40x 41  42  43  44  45  46  47  48  49x

# url = 'rtsp://admin:hk123456@192.168.1.192:554/h264/ch43/main/av_stream'
# 36  37  38  39  40  41  42  43  44  45  46  47x 48x 49x

print('Start!!!')

# 读取视频流
cap = cv2.VideoCapture(url)

print('Start OK!!!')

# 设置视频参数
# cap.set(3, 480)

print(cap.isOpened())

print(sys.version)

print(cv2.__version__)
i=0

while cap.isOpened():
    ret_flag, img_camera = cap.read()
    
    cv2.imshow("camera", img_camera)

    # 每帧数据延时 1ms, 延时为0, 读取的是静态帧
    k = cv2.waitKey(1)
    if k == ord('s'):
        i+=1
        cv2.imwrite(str(i) + '.png', img_camera)
    if k == ord('1'):
        break

# 释放所有摄像头
cap.release()

# 删除窗口
cv2.destroyAllWindows()
