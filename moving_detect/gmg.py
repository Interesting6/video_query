import cv2 
import numpy as np
from matplotlib import pyplot as plt
import os, shutil, time


videos = ['1.avi', '2.mp4', 'ch01.mp4', 'ch06.mp4', 'ch07.mp4', 'door.mp4']
result_pth = './GMG/using_time_res.txt'
elker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,13))

for video_path in videos:
    stime = time.time()
    video_path = '/home/cym/Datasets/videos/' + video_path
    video_name = video_path.split('/')[-1].replace('.', '_')
    bs = cv2.bgsegm.createBackgroundSubtractorGMG()
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(5))
    size = int(cap.get(3)), int(cap.get(4))

    sqker = np.ones((5,5),np.uint8)
    elker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 13))
    res_path = './GMG/res_' + video_name
    if os.path.exists(res_path):
        shutil.rmtree(res_path)
    os.mkdir(res_path)


    i = -1
    while cap.isOpened():
        i += 1
        ret, frame = cap.read()
        if not ret:
            print("Video end!")
            break
        if i < 120:
            fgmask = bs.apply(frame)
            continue

        if i % fps == 0:
            sec = i // fps
            fgmask = bs.apply(frame)
            bframe = cv2.threshold(fgmask.copy(), 244, 255, cv2.THRESH_BINARY)[1]
            # bframe = cv2.morphologyEx(bframe, cv2.MORPH_OPEN, sqker)
            # bframe = cv2.dilate(bframe, elker, iterations=2)
            bframe = cv2.erode(bframe, None, iterations = 1)
            bframe = cv2.dilate(bframe, elker, iterations = 2)
            image, contours, hier = cv2.findContours(bframe, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) != 0:
                det_num = 0
                for c in contours:
                    if cv2.contourArea(c) > 1000:
                        (x,y,w,h) = cv2.boundingRect(c)
                        if x<500 and y<80:
                            continue
                        det_num += 1
                        xl, yu = x - 20, y - 30
                        if x < 20:
                            xl = x - x // 2
                        if y < 30:
                            yu = y - y // 2
                        crop_img = frame[yu: y + h + 20, xl: x + w + 30] 
                        # cv2.rectangle(frame, (xl,yu), (xl+w+30, y+h+20), (255, 255, 0), 2)
                        cv2.imwrite('{}/{}s{}n.jpg'.format(res_path, sec, det_num), crop_img)
                if det_num:
                    print("{}s detected!".format(sec))

    cap.release()
    # cv2.destroyAllWindows()
    etime = time.time()
    det_time = etime - stime
    # print('Total Time: ' + str(det_time) + 's')
    with open(result_pth, 'a+', encoding='utf-8') as f:
        f.write(video_name + ': total time: ' + str(det_time) + 's \n')


