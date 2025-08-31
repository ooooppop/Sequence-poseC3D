import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt
import datetime


path = r"D:\WuShuang\mmaction2-main\tools\data\skeleton\mnist_train.pkl"
data_list = []
start_idx, end_idx = -25, -5
with open(path, 'rb') as f:
    data = pickle.load(f)
    for i in range(len(data['annotations'])):
        # data['annotations'][i]['keypoint_score'][0][start_idx:end_idx, 5:11] = 0.5

        keypoints = data['annotations'][i]['keypoint'][0]
        # 计算全局x和y的最小/最大值
        x_min, x_max = keypoints[:, 11:, 0].min(), keypoints[:, 11:, 0].max()
        y_min, y_max = keypoints[:, 11:, 1].min(), keypoints[:, 11:, 1].max()

        random_x = np.random.uniform(x_min, x_max, size=(end_idx-start_idx, 6, 1))  # x坐标随机
        random_y = np.random.uniform(y_min, y_max, size=(end_idx-start_idx, 6, 1))  # y坐标随机
        random_coords = np.concatenate([random_x, random_y], axis=2)  # 合并为(10,17,2)
        data['annotations'][i]['keypoint'][0][start_idx:end_idx, 11:, :] = random_coords


with open(r"D:\WuShuang\mmaction2-main\tools\data\skeleton\cut_rewrite_mnist_leg_new_.pkl", 'wb') as f:
    pickle.dump(data, f)

'''
0: 鼻子 (nose)
1: 左眼 (left_eye)
2: 右眼 (right_eye)
3: 左耳 (left_ear)
4: 右耳 (right_ear)
5: 左肩 (left_shoulder)
6: 右肩 (right_shoulder)
7: 左肘 (left_elbow)
8: 右肘 (right_elbow)
9: 左腕 (left_wrist)
10: 右腕 (right_wrist)
11: 左髋 (left_hip)
12: 右髋 (right_hip)
13: 左膝 (left_knee)
14: 右膝 (right_knee)
15: 左踝 (left_ankle)
16: 右踝 (right_ankle)
'''


head = [0, 1, 2, 3, 4]
trunk = [5, 6, 11, 12]
arm = [5, 6, 7, 8, 9, 10]
leg = [11, 12, 13, 14, 15, 16]

