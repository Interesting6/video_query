import cv2 
import numpy as np
# from matplotlib import pyplot as plt
import os, shutil, time
from utils4ymc import logger, check_path


def update_alpha(o_alpha, C, det=0):
    t = (1 / (C+1))**0.5
    if det == 0: # 未检测到运动物体，该帧加入背景的权重增大
        alpha = o_alpha*(1+t)
    else:
        alpha = o_alpha*(t)
    return alpha


res_stats_pth = './Results/results_stats.txt'
@logger(res_stats_pth)
def move_avg_bg(video_pth, alpha=0.1, n_dil=4, n_ero=1, thre=25, area=4096):
    video_name = video_pth.split('/')[-1].replace('.', '_')
    res_name = str(alpha).replace('.', '') + f'_d{n_dil}_t{thre}' 
    res_path = f"./Results/{video_name}/{res_name}"
    check_path(res_path)


    cap = cv2.VideoCapture(video_pth)
    fps = int(cap.get(5))
    size = w, h = int(cap.get(3)), int(cap.get(4))
    sqker = np.ones((5,5),np.uint8)

    avg = None
    C = 0
    incre = 0
    no_incre = 0
    o_alpha = alpha
    o_mask = np.zeros((h, w), dtype=bool)

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
            gframe = cv2.GaussianBlur(gframe,(5,5),0)#高斯滤波  # uint8
            if avg is None:
                avg = gframe.astype('float')
            # cv2.accumulateWeighted(gframe, avg, alpha)  # avg # uint8形式的float64类型
            frame_delta = cv2.absdiff(gframe, cv2.convertScaleAbs(avg)) # uint8
            bframe = cv2.threshold(frame_delta, thre, 255, cv2.THRESH_BINARY)[1] # uint8
            bframe = cv2.erode(bframe, sqker, iterations=n_ero)
            bframe_ = cv2.dilate(bframe, sqker, iterations=1)
            bframe = cv2.dilate(bframe_, sqker, iterations=n_dil-1)
            contours = cv2.findContours(bframe, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]

            det_num = 0
            rects = []
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
                    yslice = slice(yu, y+h+20)
                    xslice = slice(xl, x+w+12)
                    rects.append((yslice, xslice))
                    crop_img = frame[yslice, xslice]
                    cv2.imwrite(f'{res_path}/{sec}s_{det_num}o.jpg', crop_img)
            if det_num:
                # incre += 1
                # C += 1
                # C = C if C<=10 else 10 # 最大为14
                # no_incre = 0
                # alpha = update_alpha(o_alpha, C, 1)
                print(f"{sec}s detected {det_num} obj!")
            # else:
            #     no_incre += 1
            #     if no_incre == 7:
            #         C = 0
            #     alpha = update_alpha(o_alpha, C, 0)

            if det_num:
                mask2 = bframe_ == 255  # 背景运动物体部分的mask
                mask = o_mask.copy() 
                for ysl, xsl in rects:
                    mask[ysl, xsl] = 1
                mask = np.bitwise_and(mask, mask2)
                bgs = avg[~mask].copy()
                cv2.accumulateWeighted(gframe[~mask], bgs, alpha) # 背景静止部分更新
                avg[~mask] = bgs
                # cv2.accumulateWeighted(gframe[mask], avg[mask], 0.1*alpha) # 运动物体部分不更新
            else:
                cv2.accumulateWeighted(gframe, avg, alpha)
            

    cap.release()



if __name__ == "__main__":
    video_pth = "/home/cym/Datasets/videos/2.avi"
    alpha = 0.03
    n_dil = 3
    move_avg_bg(video_pth, alpha=alpha, n_dil=n_dil, thre=20)
