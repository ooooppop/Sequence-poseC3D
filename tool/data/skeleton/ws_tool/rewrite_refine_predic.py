import numpy as np
import pickle

head = [0, 1, 2, 3, 4]
trunk = [5, 6, 11, 12]
arm = [5, 6, 7, 8, 9, 10]
leg = [11, 12, 13, 14, 15, 16]

path_0 = r"D:\WuShuang\mmaction2-main\tools\data\skeleton\mnist_train.pkl"
path_1 = r"D:\WuShuang\mmaction2-main\tools\data\skeleton\predict_rewrite_mnist_25.pkl"

data_list = []
# start_idx, end_idx = -25, -15
with open(path_1, 'rb') as f:
    data_1 = pickle.load(f)
with open(path_0, 'rb') as f:
    data_0 = pickle.load(f)
for i in range(len(data_0['annotations'])):
    data_0['annotations'][i]['keypoint'][0][-25:, arm, :] = data_1['annotations'][i]['keypoint'][0][-25:, arm, :]

with open(r"D:\WuShuang\mmaction2-main\tools\data\skeleton\predict_refine_arm_25.pkl", 'wb') as f:
    pickle.dump(data_0, f)
