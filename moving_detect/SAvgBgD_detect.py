import cv2 
import numpy as np
# from matplotlib import pyplot as plt
import os, shutil, time
from functools import wraps


def log(file_path):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            stime = time.time()
            res = func(*args, **kwargs)
            etime = time.time()
            using_time = etime - stime
            print('using time: ', using_time)
            with open(file_path, 'a+') as f:
                f.write('Run func: {} using time {}s, with the parameters:\n\
                    {}, {}. \n\n'.format(func.__name__, using_time, args, kwargs))
        return wrapper
    return decorator


res_stats_pth = './Results/results_stats.txt'
@log(res_stats_pth)
def move_avg_bg(video_pth, alpha=0.1, n_dil=4, n_ero=1, thre=25, area=4096):
    video_name = video_pth.split('/')[-1].replace('.', '_')
    res_name = str(alpha).replace('.', '') + f'_{n_dil}'
    res_path = f"./Results/{video_name}/{res_name}"
    if os.path.exists(res_path):
        shutil.rmtree(res_path)
    os.makedirs(res_path)

    cap = cv2.VideoCapture(video_pth)
    fps = int(cap.get(5))
    size = int(cap.get(3)), int(cap.get(4))
    sqker = np.ones((5,5),np.uint8)

    avg = None

    i = -1
    while cap.isOpened():
        i += 1
        ret, frame = cap.read()
        if not ret:
            print("Video end!")
            break
        if i % fps == 0:
            sec = i // fps
            gframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#变成灰色图像
            gframe = cv2.GaussianBlur(gframe,(5,5),0)#高斯滤波
            if avg is None:
                avg = gframe.astype('float')
            cv2.accumulateWeighted(gframe, avg, alpha)
            frame_delta = cv2.absdiff(gframe, cv2.convertScaleAbs(avg))
            bframe = cv2.threshold(frame_delta, thre, 255, cv2.THRESH_BINARY)[1]
            bframe = cv2.erode(bframe, sqker, iterations=n_ero)
            bframe = cv2.dilate(bframe, sqker, iterations=n_dil)
            contours = cv2.findContours(bframe, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]

            det_num = 0
            for idx, c in enumerate(contours, 1):
                if cv2.contourArea(c) > area: 
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
                    cv2.imwrite(f'{res_path}/{sec}s_{det_num}o.jpg', crop_img)
            if det_num:
                print(f"{sec}s detected {det_num} obj!")

    cap.release()



if __name__ == "__main__":
    video_pth = "/home/cym/Datasets/videos/2.avi"
    alpha = 0.02
    n_dil = 4
    move_avg_bg(video_pth, alpha=alpha, n_dil=n_dil)


