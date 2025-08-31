import cv2
import matplotlib.pyplot as plt
import pickle
import numpy as np
import os

# 配置参数
VIDEO_PATH = r"D:\WuShuang\mmaction2-main\tools\data\skeleton\data\my_video\pushing\S001C002P003R002A052_rgb.avi"
PKL_PATH = r"D:\WuShuang\mmaction2-main\tools\data\skeleton\ws_tool\test_data\pkl4\S001C002P003R002A052_rgb_pre1.pkl"
# PKL_PATH_2 = None  # 第二个人的pkl路径
OUTPUT_DIR = r"D:\WuShuang\mmaction2-main\tools\data\skeleton\ws_tool\pipeline_visiua_skeleton_50_pre1"
START_FRAME = 0
END_FRAME = 50  # 最大帧数减一
UNIFIED_COLOR = True  # 是否使用统一颜色（True为统一颜色，False为分部位颜色）
OUTPUT_WIDTH = 800
OUTPUT_HEIGHT = 600
MARGIN_RATIO = 0.15
COLORS = ['#FF0000', '#00FF00']  # 两人颜色

# 骨架连接定义
CONNECTIONS = [
    (0, 1), (0, 2), (6, 8), (8, 10),
    (5, 7), (7, 9), (5, 11), (6, 12),
    (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16)
]


# 加载数据
def load_multi_person(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return {
        'person1': data['keypoint'][0].reshape(-1, 17, 2),
        'person2': data['keypoint'][1].reshape(-1, 17, 2)
    }


# 创建输出目录
output_modes = ['p1_only', 'p2_only', 'both']
for mode in output_modes:
    os.makedirs(os.path.join(OUTPUT_DIR, mode), exist_ok=True)


# 坐标变换计算
def calculate_transform(skeletons):
    valid_points = []
    for skeleton in skeletons:
        if skeleton is None:
            continue
        valid = skeleton[(skeleton >= 0).all(axis=1) &
                        (skeleton[:,0] < 1920) &
                        (skeleton[:,1] < 1080)]
        valid_points.extend(valid)

    if not valid_points:
        return 1.0, (0, 0)

    points = np.array(valid_points)
    min_coord = np.min(points, axis=0)
    max_coord = np.max(points, axis=0)

    width = max_coord[0] - min_coord[0]
    height = max_coord[1] - min_coord[1]

    scale = min(
        OUTPUT_WIDTH * (1 - MARGIN_RATIO) / max(width, 1),
        OUTPUT_HEIGHT * (1 - MARGIN_RATIO) / max(height, 1)
    )

    center = (min_coord + max_coord) / 2
    offset = np.array([OUTPUT_WIDTH/2, OUTPUT_HEIGHT/2]) - center * scale

    return scale, offset


# 可视化函数
def plot_combined(skeletons, save_path):
    plt.figure(figsize=(OUTPUT_WIDTH/100, OUTPUT_HEIGHT/100), dpi=100)
    plt.xlim(0, OUTPUT_WIDTH)
    plt.ylim(OUTPUT_HEIGHT, 0)
    plt.gca().set_aspect('equal')
    plt.axis('off')
    plt.fill_between([0, OUTPUT_WIDTH], 0, OUTPUT_HEIGHT, color='white')

    scale, offset = calculate_transform([s for s in skeletons if s is not None])

    for idx, skeleton in enumerate(skeletons):
        if skeleton is None:
            continue

        transformed = skeleton * scale + offset
        # 绘制连接线
        for (s, e) in CONNECTIONS:
            if s >= 17 or e >= 17:
                continue
            start = transformed[s]
            end = transformed[e]
            if (start < 0).any() or (start > [OUTPUT_WIDTH, OUTPUT_HEIGHT]).any():
                continue
            if (end < 0).any() or (end > [OUTPUT_WIDTH, OUTPUT_HEIGHT]).any():
                continue
            plt.plot([start[0], end[0]], [start[1], end[1]],
                    color=COLORS[idx], linewidth=2, zorder=1)

        # 绘制关键点
        for i, (x, y) in enumerate(transformed):
            if x < 0 or y < 0 or x > OUTPUT_WIDTH or y > OUTPUT_HEIGHT:
                continue
            plt.scatter(x, y, s=100, c=COLORS[idx],
                       edgecolors='black', linewidths=0.5, zorder=2)

    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

# 主处理流程
data = load_multi_person(PKL_PATH)
total_frames = min(len(data['person1']), len(data['person2']))
END_FRAME = min(END_FRAME, total_frames - 1)

for frame_idx in range(START_FRAME, END_FRAME + 1):
    # 生成三种组合
    combinations = [
        (data['person1'][frame_idx], None),          # 仅第一人
        (None, data['person2'][frame_idx]),          # 仅第二人
        (data['person1'][frame_idx], data['person2'][frame_idx])  # 两人同框
    ]

    for idx, (p1, p2) in enumerate(combinations):
        mode = output_modes[idx]
        save_dir = os.path.join(OUTPUT_DIR, mode)
        save_path = os.path.join(save_dir, f"frame_{frame_idx:04d}.png")

        skeletons = [p1, p2]  # 重要！保持列表长度为2的结构

        # 检查是否有有效数据
        if not any(s is not None and s.max() >= 0 for s in skeletons):
            continue  # 跳过全无效帧
        # # 过滤无效数据
        # skeletons = []
        # if p1 is not None and p1.max() >= 0:
        #     skeletons.append(p1)
        # if p2 is not None and p2.max() >= 0:
        #     skeletons.append(p2)

        if skeletons:
            plot_combined(skeletons, save_path)

print(f"处理完成！生成模式：{output_modes}")
print(f"总帧数：{END_FRAME - START_FRAME + 1}")