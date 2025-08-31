# show_pkl.py
import numpy as np
import pickle
import numpy
# path = r"D:\WuShuang\mmaction2-main\tools\data\skeleton\predict_rewrite_mnist.pkl"
# path = r"D:\WuShuang\mmaction2-main\tools\data\skeleton\mnist_train.pkl"
# path = r"D:\WuShuang\mmaction2-main\results\predictions.pkl"
# path = r"D:\WuShuang\mmaction2-main\tools\data\skeleton\cut_rewrite_mnist_head_new_10.pkl"
# path = r"D:\WuShuang\mmaction2-main\tools\data\skeleton\ws_tool\test_results\S001C002P002R002A052_rgb.pkl"
# path = r"D:\WuShuang\mmaction2-main\tools\data\skeleton\ws_tool\test_data\pkl4\248421290.pkl"
path = r"D:\WuShuang\mmaction2-main\tools\data\skeleton\ws_tool\test_data\pkl4\S001C002P003R002A052_rgb_new.pkl"
# path = r"D:\WuShuang\mmaction2-main\tools\data\skeleton\ws_tool\output_keypoints.pkl"
f = open(path, 'rb')
data = pickle.load(f)
# data_1 = data['annotations']
# for i in range(len(data_1)):
#     if data_1[i]["frame_dir"]=='S002C003P010R002A052_rgb':
#         select_data = data_1[i]
#         print(i)  # 737
# first_frame_result = data[1]
# keypoints = first_frame_result.keypoints.data  # 形状为 (num_persons, num_keypoints, 3)
a = data['keypoint']
b = data['keypoint_score']
# a = data['keypoint_score'].reshape(np.shape(data['keypoint_score'])[1], 17)
# b = data['keypoint'].reshape(np.shape(data['keypoint_score'])[1], 17, 2)
# print(a)
# print(len(data))
print(data)
