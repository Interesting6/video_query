import cv2 
import numpy as np
# from matplotlib import pyplot as plt
import os, shutil, time


stime = time.time()
bs = cv2.bgsegm.createBackgroundSubtractorGMG()
video_pth = "/root/Datasets/videos/2.avi"
video_name = video_pth.split('/')[-1].replace('.', '_')
cap = cv2.VideoCapture(video_pth)
fps = int(cap.get(5))
size = int(cap.get(3)), int(cap.get(4))

sqker = np.ones((5,5),np.uint8)
elker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 13))
res_path = './AvgBG/{}'.format(video_name) + '_05'
res_stats_pth = './AvgBG/results_stats.txt'
if os.path.exists(res_path):
    shutil.rmtree(res_path)
os.mkdir(res_path)

stop_fi = 5*60*fps
avg = None

i = -1
while cap.isOpened():
    i += 1
    ret, frame = cap.read()
    if not ret: # or i>stop_fi:
        print("Video 5min end!")
        break
    if i % fps == 0:
        sec = i // fps
        gframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#变成灰色图像
        # gframe = cv2.GaussianBlur(gframe,(5,5),0)#高斯滤波
        if avg is None:
            avg = gframe.astype('float')
        cv2.accumulateWeighted(gframe, avg, 0.05)
        frame_delta = cv2.absdiff(gframe, cv2.convertScaleAbs(avg))
        bframe = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        bframe = cv2.erode(bframe, sqker, iterations=1)
        bframe = cv2.dilate(bframe, elker, iterations=2)
        contours, hier = cv2.findContours(bframe, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        det_num = 0
        for idx, c in enumerate(contours, 1):
            if cv2.contourArea(c) > 1000: 
                (x,y,w,h) = cv2.boundingRect(c)
                if x<800 and y<90:
                    continue
                det_num += 1
                xl, yu = x - 12, y - 20
                if x < 12:
                    xl = x - x // 2
                if y < 20:
                    yu = y - y // 2
                crop_img = frame[yu: y + h + 20, xl: x + w + 12] 
                cv2.imwrite('{}/{}s_{}f.jpg'.format(res_path, sec, idx), crop_img)
        if det_num:
            print("{}s detected {} obj!".format(sec, det_num))




cap.release()
det_time = time.time()-stime
print('using time: ', det_time)
with open(res_stats_pth, 'a+') as f:
    f.write(video_name + ': total time: ' + str(det_time) + 's \n')







