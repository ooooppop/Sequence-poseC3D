import cv2
import matplotlib.pyplot as plt
import pickle
import numpy as np

# 加载pkl文件
# pkl_path = r"D:\WuShuang\mmaction2-main\tools\data\skeleton\mnist_train.pkl"
# pkl_path = r"D:\WuShuang\mmaction2-main\tools\data\skeleton\ws_tool\test_results\output.pkl"
# pkl_path = r"D:\WuShuang\mmaction2-main\tools\data\skeleton\ws_tool\test_data\pkl4\S001C002P002R002A052_rgb.pkl"
pkl_path = r"D:\WuShuang\mmaction2-main\tools\data\skeleton\ws_tool\test_data\pkl4\S001C002P003R002A052_rgb_new.pkl"
with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

# 取第一个标注数据
# anno = data['annotations'][0]
anno = data

# 视频路径（实际应是被处理的原视频路径）
# video_path = r"D:\WuShuang\mmaction2-main\tools\data\skeleton\data\my_video\walking\001-bg-02-036.avi"
# video_path = r"D:\WuShuang\mmaction2-main\tools\data\skeleton\ws_tool\test_data\pushing_1\S001C002P002R002A052_rgb.avi"
video_path = r"D:\WuShuang\mmaction2-main\tools\data\skeleton\data\my_video\pushing\S001C002P003R002A052_rgb.avi"

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

def visualize_skeleton(frame, keypoints):
    """ 可视化骨架的函数 """
    connections = [
        (0, 1), (0, 2), (6, 8), (8, 10),
        (5, 7), (7, 9), (5, 11), (6, 12),
        (11, 12), (11, 13), (13, 15),
        (12, 14), (14, 16)
    ]

    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    for i, (x, y) in enumerate(keypoints):
        plt.scatter(x, y, c='red' if i in [5, 7, 9, 11, 13, 15] else 'blue')
        plt.text(x, y, str(i), color='white', fontsize=8)

    for (s, e) in connections:
        start = keypoints[s]
        end = keypoints[e]
        plt.plot([start[0], end[0]], [start[1], end[1]], 'y-')
    plt.show()


# 从视频中读取第一帧
cap = cv2.VideoCapture(video_path)
success, frame = cap.read()
cap.release()

if success:
    # 获取第一个人的第一帧关键点
    keypoints = anno['keypoint'].reshape(116, 17, 2)[40]  # 形状应为 (17, 2)

    # 可视化
    visualize_skeleton(frame, keypoints)
else:
    print(f"无法读取视频: {video_path}")
