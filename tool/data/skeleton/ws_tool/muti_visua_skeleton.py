import os
import cv2
import matplotlib.pyplot as plt
import pickle
import numpy as np

# 配置参数
pkl_path_1 = r"D:\WuShuang\mmaction2-main\tools\data\skeleton\ws_tool\test_data\pkl4\S001C002P003R002A052_rgb_pre1.pkl"
pkl_path = r"D:\WuShuang\mmaction2-main\tools\data\skeleton\ws_tool\test_data\pkl4\S001C002P003R002A052_rgb_new.pkl"
video_path = r"D:\WuShuang\mmaction2-main\tools\data\skeleton\ws_tool\test_data\pushing_2\S001C002P003R002A052_rgb.avi"
output_dir = r"D:\WuShuang\mmaction2-main\tools\data\skeleton\ws_tool\processed_frames_3"  # 输出目录
frame_limit = None  # 设为None处理全部帧，或设置数字限制处理帧数

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 加载骨架数据
with open(pkl_path, 'rb') as f:
    anno = pickle.load(f)['keypoint']

with open(pkl_path_1, 'rb') as f:
    anno_1 = pickle.load(f)['keypoint']
# 初始化视频捕获
cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
valid_frames = min(total_frames, anno.shape[1])  # 取视频和骨架数据的最小帧数
if frame_limit:
    valid_frames = min(valid_frames, frame_limit)

# 骨架连接定义
CONNECTIONS = [
    (0, 1), (0, 2), (6, 8), (8, 10),
    (5, 7), (7, 9), (5, 11), (6, 12),
    (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16)
]


def plot_skeleton(frame, skeletons, save_path):
    """ 快速绘制并保存骨架图 """
    plt.figure(figsize=(12, 7))
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    for pid, kpts in enumerate(skeletons):
        color = 'cyan' if pid % 3 == 2 else 'white'

        # 绘制关节点
        for j, (x, y) in enumerate(kpts):
            plt.scatter(x, y, c=color, s=40, edgecolors='white', alpha=0.7)
            # plt.text(x + 5, y + 5, str(j), color='white', fontsize=10, alpha=0.8)

        # 绘制连接线
        for s, e in CONNECTIONS:
            plt.plot([kpts[s][0], kpts[e][0]],
                     [kpts[s][1], kpts[e][1]],
                     color=color, linewidth=2.5, alpha=0.7)

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=200)
    plt.close()


# 批量处理
for frame_idx in range(valid_frames):
    # 读取视频帧
    ret, frame = cap.read()
    if not ret:
        print(f"帧 {frame_idx} 读取失败")
        continue

    # 提取骨架数据（假设最多2人）
    skeletons = [anno[0, frame_idx], anno[1, frame_idx], anno_1[1, frame_idx]]  # 根据实际数据维度调整

    # 生成保存路径
    output_path = os.path.join(output_dir, f"frame_{frame_idx:04d}.jpg")

    # 绘制并保存
    plot_skeleton(frame, skeletons, output_path)

    # 进度显示
    if frame_idx % 1 == 0:
        print(f"已处理 {frame_idx + 1}/{valid_frames} 帧")

cap.release()
print(f"处理完成！共保存 {valid_frames} 帧到 {output_dir}")
