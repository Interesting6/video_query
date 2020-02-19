# -*- coding:utf-8 -*-
# Author: Treamy

import numpy as np
import cv2
import os, shutil



def calc_interval(obj_occ_sec, stay_t=5, false_t=2, addition_t=2):
    """
        用于将出现查询物体的时间点转化为每个小区间
        obj_occ_sec, ndarray, 出现了要查询物体的时间点集
        stay_t, int  default 5, 停留的时间阈值
        false_t, int default 2, 突然出现query_obj的连续时间是正常的阈值，连续时间小于这个阈值被认为是误分类的，如连续时间为1
        addition_t, int default 2,  给视频收尾增加额外时间
    """
    assert isinstance(obj_occ_sec, np.ndarray), "input obj_occ_sec must be np.ndarray datatype"

    diff = np.diff(obj_occ_sec)
    stop_idx = np.where(diff>stay_t)[0]
    stop = obj_occ_sec[stop_idx]
    stop = np.append(stop, obj_occ_sec[-1]) + 1 # 左闭右开

    start = obj_occ_sec[stop_idx+1]
    start = np.append(obj_occ_sec[0], start)

    start = start - addition_t
    stop = stop + addition_t

    inter_ = stop - start
    idxs = inter_ > false_t
    interval = np.stack(list(zip(start[idxs], stop[idxs])))

    return interval


def extract(obj_occ_sec, vid_pth, vid_res_dir, query_obj):
    """
        
        obj_occ_sec: array-like, sorted query object occuring time in video 
        vid_pth: original video path
        vid_res_dir: str, the results video fragmen restore path
        query_obj: str, the name of query object
    """
    interval = calc_interval(obj_occ_sec)

    if os.path.exists(vid_res_dir):
        shutil.rmtree(vid_res_dir)
    os.mkdir(vid_res_dir)

    vid_pth = os.path.expanduser(vid_pth)
    
    cap = cv2.VideoCapture(vid_pth)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    W, H = cap.get(3), cap.get(4)
    size = (int(W), int(H))


    i = -1
    for idx, inter in enumerate(interval, 1):
        store_pth = '{}{}_res{}_{}to{}.avi'.format(vid_res_dir, query_obj, idx, inter[0], inter[1])
        vidWriter = cv2.VideoWriter(store_pth, cv2.VideoWriter_fourcc(*'XVID'), fps, size)
        left, right = inter * fps
        while cap.isOpened():
            i += 1
            ret, frame = cap.read()
            if not ret:
                break
            if i >= left:
                if i < right:
                    vidWriter.write(frame)
                else:
                    break


    vidWriter.release()
    cap.release()






if __name__ == "__main__":
    obj_occ_sec = [1,2,3,4,5,6,7,8, 15,17,19,20,21, 30,31,32,34,35, 40,41, 50,51, 60, 70,71,72, 80,81,82,83]
    obj_occ_sec = np.array(obj_occ_sec)
    interval = calc_interval(obj_occ_sec)
    print(interval)
    # 不前后扩充2秒的结果为：
    # [[ 1  9]
    # [15 22]
    # [30 42]
    # [70 73]
    # [80 84]]


    vid_pth = '~/Datasets/videos/2.avi'
    vid_res_dir = './Test/extract/'
    query_obj = 'man'
    extract(obj_occ_sec, vid_pth, vid_res_dir, query_obj)


