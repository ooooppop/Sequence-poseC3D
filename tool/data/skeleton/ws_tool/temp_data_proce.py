import os
import cv2
import matplotlib.pyplot as plt
import pickle
import numpy as np
import random


# 配置参数
pkl_path = r"D:\WuShuang\mmaction2-main\tools\data\skeleton\ws_tool\test_data\pkl4\S001C002P003R002A052_rgb_new.pkl"

with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

error = [19, 20, 24, 25, 26, 39, 40, 41, 42, 48, 49, 50, 51, 52]
error_keypoint = [8, 10]
# for i in error:
#     temp = data['keypoint'][0, i, :, :].copy()
#     data['keypoint'][0, i, :, :] = data['keypoint'][1, i, :, :]
#     data['keypoint'][1, i, :, :] = temp

for i in range(33, 56):
    for j in range(17):
        if j == 8:
            data['keypoint'][1, i, j, 0] = data['keypoint'][1, i-1, j, 0] + 3
            data['keypoint'][1, i, j, 1] = data['keypoint'][1, i-1, j, 1] + 2
        elif j == 10:
            data['keypoint'][1, i, j, 0] = data['keypoint'][1, i-1, j, 0] + 1
            data['keypoint'][1, i, j, 1] = data['keypoint'][1, i-1, j, 1] + 1
        else:
            data['keypoint'][1, i, j, 0] = data['keypoint'][1, i, j, 0] + random.uniform(-5, 5)
            data['keypoint'][1, i, j, 1] = data['keypoint'][1, i, j, 1] + random.uniform(-5, 5)


with open(r"D:\WuShuang\mmaction2-main\tools\data\skeleton\ws_tool\test_data\pkl4\S001C002P003R002A052_rgb_pre1.pkl", 'wb') as f:
    pickle.dump(data, f)