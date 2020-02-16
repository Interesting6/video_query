import time, os, shutil
import numpy as np
import cv2


def detect(pack):
    video_time = pack[0]
    frame1 = pack[1]
    frame2 = pack[2]
    frame3 = pack[3]
    det_store_pth = pack[4]

    frameDelta1 = cv2.absdiff(frame1, frame2)
    frameDelta2 = cv2.absdiff(frame2, frame3)

    thresh = cv2.bitwise_and(frameDelta1, frameDelta2) 
    thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(thresh, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=3)
    thresh = cv2.erode(thresh, None, iterations=1)
    
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]

    det_num = 0
    for idx, c in enumerate(contours, 1):
        area = cv2.contourArea(c)
        if area > 1024:
            det_num += 1
            (x, y, w, h) = cv2.boundingRect(c)
            if x<500 and y<80:
                continue
            xl, yu = x - 10, y - 20
            if x < 10:
                xl = x - x // 2
            if y < 20:
                yu = y - y // 2
            crop_img = frame[yu: y + h + 10, xl: x + w + 20] 
            cv2.imwrite("{}/{}s_{}f.jpg".format(det_store_pth, video_time, det_num), crop_img)
    if det_num:
        print("{}s detected {} obj!".format(video_time, det_num))
           

videos = ['1.avi', '2.mp4', 'ch01.mp4', 'ch06.mp4', 'ch07.mp4', 'door.mp4']
result_pth = './FDiff/using_time_res.txt'
for videoPath in videos:
    stime = time.time()
    # videoPath = '/home/cym/Datasets/videos/2.avi'
    videoPath = '/home/cym/Datasets/videos/' + videoPath
    video_name = videoPath.split('/')[-1].replace('.', '_')
    store_pth = './FDiff/' + video_name
    if os.path.exists(store_pth):
        shutil.rmtree(store_pth)
    os.mkdir(store_pth)


    cap = cv2.VideoCapture(videoPath)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print('Video FPS: ', fps)
    queryInterval = 1
    num = fps * queryInterval

    i=-1
    while cap.isOpened():
        i=i+1
        (ret, frame) = cap.read()

        if not ret:
            break

        if i%num == 0:
            frame1 = frame

        if i%num == 5:
            frame2 = frame

        if i%num == 10:
            frame3 = frame

            t = i // fps
            print('Video Time: ', t)
            pack = [t, frame1,frame2,frame3, store_pth]
            detect(pack)


    etime = time.time()
    det_time = etime - stime
    # print('Total Time: ' + str(det_time) + 's')
    with open(result_pth, 'a+', encoding='utf-8') as f:
        f.write(video_name + ': total time: ' + str(det_time) + 's \n')



