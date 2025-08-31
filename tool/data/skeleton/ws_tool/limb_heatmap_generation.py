import numpy as np
import matplotlib.pyplot as plt
import mmengine
import os
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LinearSegmentedColormap
from collections import defaultdict

# 自定义色谱
colors = [(0.18, 0.00, 0.31),  # 深紫 #2E004F
          (0.0, 1.0, 0.0),     # 纯绿 #00FF00
          (1.0, 1.0, 0.0)]     # 纯黄 #FFFF00
custom_cmap = LinearSegmentedColormap.from_list("limb_green", colors)
# 定义肢体连接关系（基于COCO 17关键点格式）
LIMB_CONNECTIONS = [
    (0, 1),   # 鼻子 -> 左眼
    (0, 2),   # 鼻子 -> 右眼
    (1, 3),   # 左眼 -> 左耳
    (2, 4),   # 右眼 -> 右耳
    (5, 6),   # 左肩 -> 右肩
    (5, 7),   # 左肩 -> 左肘
    (7, 9),   # 左肘 -> 左腕
    (6, 8),   # 右肩 -> 右肘
    (8, 10),  # 右肘 -> 右腕
    (11, 12), # 左髋 -> 右髋
    (5, 11),  # 左肩 -> 左髋
    (6, 12),  # 右肩 -> 右髋
    (11, 13), # 左髋 -> 左膝
    (13, 15), # 左膝 -> 左脚踝
    (12, 14), # 右髋 -> 右膝
    (14, 16)  # 右膝 -> 右脚踝
]

def generate_limb_heatmaps(kp_coords, kp_scores,
                          frame_idx=0,
                          person_idx=0,
                          heatmap_size=(128, 128),
                          sigma=2.0,  # 肢体需要更大的高斯模糊
                          original_size=(1920, 1080)):
    """生成肢体连接热图（支持多人多帧）"""
    # 输入验证
    assert kp_coords.shape[:3] == kp_scores.shape[:3], "坐标和分数维度不匹配"

    # 处理帧索引
    if isinstance(frame_idx, int):
        frame_indices = [frame_idx]
    elif isinstance(frame_idx, (list, tuple)):
        frame_indices = list(range(frame_idx[0], frame_idx[1] + 1))
    else:
        raise TypeError("frame_idx应为int或列表/元组范围")

    # 处理人物索引
    person_indices = [person_idx] if isinstance(person_idx, int) else person_idx

    # 坐标归一化（注意原图尺寸应为 (width, height)）
    scale_factor = np.array([
        heatmap_size[1] / original_size[1],  # x方向缩放（宽→宽）
        heatmap_size[0] / original_size[0]    # y方向缩放（高→高）
    ])

    heatmaps = {}
    for f_idx in frame_indices:
        for p_idx in person_indices:
            # 初始化热图 (H, W) 单通道肢体热图
            limb_hm = np.zeros(heatmap_size, dtype=np.float32)

            # 获取当前数据（确保维度顺序正确）
            coords = kp_coords[p_idx, f_idx]  # (17, 2)
            scores = kp_scores[p_idx, f_idx]  # (17,)

            # 遍历所有预定义的肢体连接
            for (j1, j2) in LIMB_CONNECTIONS:
                # 跳过无效连接（任一节点分数低于阈值）
                if scores[j1] < 0.1 or scores[j2] < 0.1:
                    continue

                # 坐标转换（原图坐标系→热图坐标系）
                x1, y1 = coords[j1] * scale_factor
                x2, y2 = coords[j2] * scale_factor

                # 生成线段坐标
                num_points = int(np.linalg.norm([x2-x1, y2-y1]) * 2)  # 采样密度
                x_values = np.linspace(x1, x2, num_points)
                y_values = np.linspace(y1, y2, num_points)

                # 在热图上绘制线段
                for x, y in zip(x_values, y_values):
                    xi, yi = int(np.round(x)), int(np.round(y))
                    if 0 <= xi < heatmap_size[1] and 0 <= yi < heatmap_size[0]:
                        # 使用两个节点的平均分数
                        limb_hm[yi, xi] = max(limb_hm[yi, xi], (scores[j1] + scores[j2])/2)

            # 应用高斯模糊
            limb_hm = gaussian_filter(limb_hm, sigma=sigma)
            heatmaps[(f_idx, p_idx)] = limb_hm

    return heatmaps

def visualize_limb_heatmaps(heatmaps, output_path='./limb_heatmaps',
                           vmax_scale=0.8):
    """可视化肢体热图（支持多人合并显示）"""
    os.makedirs(output_path, exist_ok=True)

    # 分组数据结构：{frame: [(person_idx, heatmap_data)]}
    frame_groups = defaultdict(list)
    for (f_idx, p_idx), hm_data in heatmaps.items():
        frame_groups[f_idx].append((p_idx, hm_data))

    # 可视化单人及合并热图
    for f_idx, people_data in frame_groups.items():
        # 创建帧目录
        frame_dir = f"{output_path}/frame_{f_idx:04d}"
        os.makedirs(frame_dir, exist_ok=True)

        # 单人热图可视化
        for p_idx, hm_data in people_data:
            plt.figure(figsize=(8, 6))
            plt.imshow(hm_data, cmap='hot', vmax=np.max(hm_data)*vmax_scale)
            plt.title(f'Frame {f_idx} Person {p_idx} - Limbs')
            plt.colorbar()
            plt.axis('off')
            plt.savefig(f"{frame_dir}/person_{p_idx:02d}.png",
                       bbox_inches='tight', dpi=120)
            plt.close()

        # 多人合并热图
        if len(people_data) > 1:
            combined_hm = np.zeros_like(people_data[0][1])
            for _, hm_data in people_data:
                combined_hm = np.maximum(combined_hm, hm_data)  # 取最大值叠加

            plt.figure(figsize=(10, 8))
            plt.imshow(combined_hm, cmap='viridis',
                      vmax=np.max(combined_hm)*vmax_scale)
            plt.title(f'Frame {f_idx} - Combined Limbs')
            plt.colorbar()
            plt.axis('off')
            plt.savefig(f"{frame_dir}/combined.png",
                       bbox_inches='tight', dpi=150)
            plt.close()
# plt.imshow(combined_heatmap, cmap='viridis', vmax=vmax)
#         plt.title(f'Frame {f_idx} - Combined Heatmap ({len(people_data)} persons)')
#         plt.colorbar()
#         plt.axis('off')
#         plt.savefig(f"{combined_dir}/combined.png", bbox_inches='tight', dpi=150)
#         plt.close()
if __name__ == "__main__":
    # 加载示例数据
    path = r"D:\WuShuang\mmaction2-main\tools\data\skeleton\ws_tool\test_data\pkl4\S001C002P003R002A052_rgb_new.pkl"
    data = mmengine.load(path)

    # 维度顺序确认：N_person x T x 17 x 2
    kp = data['keypoint']
    scores = data['keypoint_score']
    original_resolution = (data['img_shape'][1], data['img_shape'][0])  # (width, height)

    # 生成肢体热图（示例：第2帧，所有人物）
    limb_heatmaps = generate_limb_heatmaps(
        kp_coords=kp,
        kp_scores=scores,
        frame_idx=0,
        person_idx=range(kp.shape[0]),  # 所有人物
        heatmap_size=(128, 128),
        sigma=1,
        original_size=original_resolution
    )

    # 可视化保存
    visualize_limb_heatmaps(
        limb_heatmaps,
        output_path='./limb_heatmaps_output',
        vmax_scale=0.7
    )