import numpy as np
import cv2
from matplotlib import pyplot as plt
import os, time


stime = time.time()
vid_path = "/home/cym/Datasets/videos/2.avi"
vid_name = vid_path.split('/')[-1].split('.')[0]
bg_store_pth = "./BgDiff/bgs/"
bg_store_pth = bg_store_pth + 'file_' + vid_name + "/"
diff_store_pth = "./BgDiff/avg_absdiffs/"
diff_store_pth = diff_store_pth + 'file_' + vid_name + "/"
det_store_pth = "./BgDiff/detect_res/"
det_store_pth = det_store_pth + 'file_' + vid_name + "/"
if not os.path.exists(det_store_pth):
    os.makedirs(det_store_pth)

cap = cv2.VideoCapture(vid_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
cap_length = int(cap.get(7)) # CV_CAP_PROP_FRAME_COUNT
W, H = cap.get(3), cap.get(4)
print("video FPS: ", fps)
scene_f = 10*60 # 一个场景为10分钟，每10分钟生成一个背景


bg_store_pth = bg_store_pth + "bg_{}m.png"
diff_store_pth = diff_store_pth + "diff_{}m.png"

scene_num = 1
avg_gframe = cv2.imread(bg_store_pth.format(scene_num), 0)
avg_absdiff = cv2.imread(diff_store_pth.format(scene_num), 0)

sqker = np.ones((5,5), np.uint8)
elker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 13))

thresh = 25
frame_num = 0
i = -1
while cap.isOpened():
    i += 1
    ret, frame = cap.read()
    if not ret:
        break

    if i % fps == 0: # 每秒测一张
        sec = i // fps
        if frame_num <= scene_f:
            frame_num += 1
            gframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_delta = cv2.absdiff(gframe, avg_gframe)
            frame_delta = cv2.absdiff(frame_delta, avg_absdiff)
            bframe = cv2.threshold(frame_delta, thresh, 255, cv2.THRESH_BINARY)[1]
            bframe = cv2.erode(bframe, sqker, iterations=1)
            bframe = cv2.dilate(bframe, elker, iterations=2)
            contours = cv2.findContours(bframe, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
            for idx, c in enumerate(contours, 1):
                area = cv2.contourArea(c)
                if  area > 1000:
                    (x,y,w,h) = cv2.boundingRect(c)
                    if x<500 and y<80:
                        continue
                    xl, yu = x - 10, y - 20
                    if x < 10:
                        xl = x - x // 2
                    if y < 20:
                        yu = y - y // 2
                    crop_img = frame[yu: y + h + 10, xl: x + w + 20] 
                    cv2.imwrite("{}{}s_{}f.jpg".format(det_store_pth, sec, idx), crop_img)
                    print("detected!")
           
        else:
            frame_num = 0
            scene_num += 1
            avg_gframe = cv2.imread(bg_store_pth.format(scene_num), 0)
            avg_absdiff = cv2.imread(diff_store_pth.format(scene_num), 0)


cap.release()
print("detect time in video {}.avi using: {}".format(vid_name, time.time()-stime))
