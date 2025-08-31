import torch
import cv2 as cv
import numpy as np
from ultralytics.data.augment import LetterBox
from ultralytics.utils import ops
from ultralytics.engine.results import Results
import pickle

# 视频路径
video_path = r"D:\WuShuang\mmaction2-main\tools\data\skeleton\ws_tool\test_data\pushing_1\S001C002P002R002A052_rgb.avi"
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
conf = 0.25
iou = 0.7

# 初始化视频捕获
cap = cv.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: 无法打开视频文件")
    exit()

# 加载YOLOv8姿态估计模型
ckpt = torch.load('yolov8n-pose.pt', map_location='cpu')
model = ckpt['model'].to(device).float()
model.eval()

# 存储所有帧的关键点数据
all_frames_data = []

frame_count = 0  # 帧计数器
while True:
    ret, frame = cap.read()
    if not ret:
        break  # 视频结束

    # ---------------------- 数据预处理 ----------------------
    # LetterBox缩放 (保持长宽比)
    processed_img = LetterBox([640, 640], auto=True, stride=32)(image=frame)

    # 转换为模型输入格式 (BCHW)
    img_array = np.array(processed_img)  # HWC格式
    img_array = img_array[..., ::-1].transpose(2, 0, 1)  # BGR转RGB + HWC转CHW
    img_tensor = torch.from_numpy(img_array).to(device)
    img_tensor = img_tensor.float().div(255).unsqueeze(0)  # 添加批次维度

    # ---------------------- 模型推理 ----------------------
    with torch.no_grad():
        predictions = model(img_tensor)

    # 非极大值抑制
    pred = ops.non_max_suppression(
        predictions,
        conf_thres=conf,
        iou_thres=iou,
        max_det=100  # 每帧最大检测人数
    )[0]  # 取第一个（唯一）批次的预测结果

    # ---------------------- 数据处理 ----------------------
    if pred is not None and len(pred) > 0:
        # 缩放边界框到原始图像尺寸
        pred[:, :4] = ops.scale_boxes(img_tensor.shape[2:], pred[:, :4], frame.shape).round()

        # 处理关键点 (形状: [num_persons, 17, 3])
        keypoints = pred[:, 6:].view(-1, *model.kpt_shape)  # 重塑为[num_persons, 17, 3]
        keypoints = ops.scale_coords(img_tensor.shape[2:], keypoints, frame.shape)

        # 转换为CPU的NumPy数组
        keypoints_np = keypoints.cpu().numpy()
        boxes_np = pred[:, :6].cpu().numpy()  # [x1, y1, x2, y2, conf, class]

        # 存储当前帧数据
        frame_data = {
            "frame_num": frame_count,
            "boxes": boxes_np,  # 形状 [num_persons, 6]
            "keypoints": keypoints_np  # 形状 [num_persons, 17, 3]
        }
        all_frames_data.append(frame_data)

    frame_count += 1
    print(f"已处理帧: {frame_count}", end='\r')  # 实时进度显示

# 释放资源
cap.release()
print(f"\n视频处理完成，总帧数: {frame_count}")

# 保存所有帧数据到文件
with open('all_keypoints.pkl', 'wb') as f:
    pickle.dump(all_frames_data, f)

print("关键点数据已保存至 all_keypoints.pkl")