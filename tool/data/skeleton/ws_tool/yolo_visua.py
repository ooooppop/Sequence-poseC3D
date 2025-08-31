import pickle
import cv2

# 配置参数
video_path = r"D:\WuShuang\mmaction2-main\tools\data\skeleton\ws_tool\test_data\pushing_1\S001C002P002R002A052_rgb.avi"
pkl_path = r"D:\WuShuang\mmaction2-main\tools\data\skeleton\ws_tool\output_keypoints.pkl"
target_frame = 0        # 要可视化的帧序号

# 1. 加载关键点数据
with open(pkl_path, 'rb') as f:
    all_frames_data = pickle.load(f)


# 2. 获取原始视频帧
def get_video_frame(video_path, frame_num):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


original_frame = get_video_frame(video_path, target_frame)
assert original_frame is not None, f"无法读取第 {target_frame} 帧"


# 3. 可视化函数
def draw_human_pose(frame, keypoints, conf_threshold=0.3):
    # 正确的骨架连接关系 (使用0-based索引)
    skeleton = [
        # 头部
        (16, 14),  # 右踝 -> 右膝
        (14, 12),  # 右膝 -> 右髋
        (17, 15),  # 左踝 -> 左膝
        (15, 13),  # 左膝 -> 左髋
        # 躯干
        (12, 13),  # 右髋 -> 左髋
        (6, 12),  # 右肩 -> 右髋
        (7, 13),  # 左肩 -> 左髋
        # 手臂
        (6, 8),  # 右肩 -> 右肘
        (8, 10),  # 右肘 -> 右手腕
        (7, 9),  # 左肩 -> 左肘
        (9, 11),  # 左肘 -> 左手腕
        # 下半身
        (2, 3),  # 右髋 -> 左髋 (原COCO定义需要修正)
        (1, 2),  # 鼻子 -> 右眼
        (1, 3),  # 鼻子 -> 左眼
        (2, 4),  # 右眼 -> 右耳
        (3, 5)  # 左眼 -> 左耳
    ]

    for kpts in keypoints:
        # 绘制关键点
        for kpt in kpts:
            x, y, conf = kpt
            if conf > conf_threshold:
                cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)

        # 绘制骨架连线
        for (s, e) in skeleton:
            s -= 1
            e -= 1
            x1, y1, c1 = kpts[s]
            x2, y2, c2 = kpts[e]
            if c1 > conf_threshold and c2 > conf_threshold:
                cv2.line(frame,
                         (int(x1), int(y1)),
                         (int(x2), int(y2)),
                         (0, 255, 255), 2)
    return frame


# 4. 执行可视化
frame_data = all_frames_data[target_frame]
visualized_img = draw_human_pose(original_frame.copy(),
                                 frame_data.keypoints.data)

# 5. 显示结果
cv2.imshow('Pose Visualization', visualized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()