import numpy as np
import cv2
from matplotlib import pyplot as plt
import os, time

"""
    background generate
"""


stime = time.time()
vid_path = "/home/cym/Datasets/videos/2.avi"
vid_name = vid_path.split('/')[-1].split('.')[0]
bg_store_pth = "./BgDiff/bgs/"
bg_store_pth = bg_store_pth + 'file_' + vid_name + "/"
diff_store_pth = "./BgDiff/avg_absdiffs/"
diff_store_pth = diff_store_pth + 'file_' + vid_name + "/"

if not os.path.exists(bg_store_pth):
    os.makedirs(bg_store_pth)
if not os.path.exists(diff_store_pth):
    os.makedirs(diff_store_pth)

cap = cv2.VideoCapture(vid_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
cap_length = int(cap.get(7)) # CV_CAP_PROP_FRAME_COUNT
W, H = cap.get(3), cap.get(4)
print("video FPS: ", fps)
scene_f = 10*60 # 一个场景为10分钟，每10分钟生成一个背景


bg_store_pth = bg_store_pth + "bg_{}m.png"
diff_store_pth = diff_store_pth + "diff_{}m.png"

time_scene_bef = time.time()
scene_num = 0 # 场景的index数
scene_gframes = []  # 一个场景存储的图片
frame_num = 0 # 一个场景中的帧数
i = -1
while cap.isOpened():
    i += 1
    ret, frame = cap.read()
    if not ret:
        break

    if i % fps == 0: # 每秒测一张
        if frame_num <= scene_f:
            frame_num += 1
            gframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            scene_gframes.append(gframe)
        else:
            scene_gframes = np.stack(scene_gframes, axis=0)
            scene_gframes = scene_gframes.astype(dtype='float16') / 255.
            avg_gframe = np.mean(scene_gframes, axis=0)
            absdiff = abs(scene_gframes - avg_gframe)
            avg_absdiff = np.mean(absdiff, axis=0)

            avg_gframe = (avg_gframe * 255).astype('uint8')
            avg_absdiff = (avg_absdiff * 255).astype('uint8')

            scene_num += 1
            cv2.imwrite(bg_store_pth.format(scene_num), avg_gframe)
            cv2.imwrite(diff_store_pth.format(scene_num), avg_absdiff)

            time_scene_aft = time.time()
            print("one scene generation end, using time: ", time_scene_aft-time_scene_bef)
            time_scene_bef = time_scene_aft

            frame_num = 0
            scene_gframes = []


if scene_gframes != []:
    scene_gframes = np.stack(scene_gframes, axis=0)
    scene_gframes = scene_gframes.astype(dtype='float16') / 255.
    avg_gframe = np.mean(scene_gframes, axis=0)
    absdiff = abs(scene_gframes - avg_gframe)
    avg_absdiff = np.mean(absdiff, axis=0)
    avg_gframe = (avg_gframe * 255).astype('uint8')
    avg_absdiff = (avg_absdiff * 255).astype('uint8')

    scene_num += 1
    cv2.imwrite(bg_store_pth.format(scene_num), avg_gframe)
    cv2.imwrite(diff_store_pth.format(scene_num), avg_absdiff)

    time_scene_aft = time.time()
    print("one scene generation end, using time: ", time_scene_aft-time_scene_bef)
    time_scene_bef = time_scene_aft

print("Video end!")
cap.release()
print("offline background and average absdiff generation time: ", time.time()-stime)




