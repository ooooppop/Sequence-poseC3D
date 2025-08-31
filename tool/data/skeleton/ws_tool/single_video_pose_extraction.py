# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
import cv2
from tempfile import TemporaryDirectory
import mmengine
import numpy as np
import torch

# 核心功能导入
from mmaction.apis import pose_inference
from mmaction.utils import frame_extract


class ModelConfig:
    """硬编码模型配置参数"""

    def __init__(self):
        # 检测模型配置
        self.det_config = 'D:/WuShuang/mmaction2-main/demo/demo_configs/faster-rcnn_r50-caffe_fpn_ms-1x_coco-person.py'
        self.det_checkpoint = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco-person/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth'
        self.det_score_thr = 0.5

        # 姿态估计模型配置
        self.pose_config = 'D:/WuShuang/mmaction2-main/demo/demo_configs/td-hm_hrnet-w32_8xb64-210e_coco-256x192_infer.py'
        self.pose_checkpoint = 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth'


def validate_pose_data(pose_data, frame_idx):
    """验证姿态数据结构"""
    if not isinstance(pose_data, dict):
        print(f"警告：第{frame_idx}帧姿态数据非字典类型，已跳过")
        return False
    required_keys = ['keypoints', 'keypoint_scores']
    missing_keys = [k for k in required_keys if k not in pose_data]
    if missing_keys:
        print(f"警告：第{frame_idx}帧缺失关键字段{missing_keys}，已跳过")
        return False
    return True


def ntu_pose_extraction(video_path, label):
    """核心处理函数"""
    cfg = ModelConfig()
    tmp_dir = TemporaryDirectory()

    # 步骤1: 视频抽帧
    frame_paths, _ = frame_extract(video_path, out_dir=tmp_dir.name)
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # 步骤2: 目标检测
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    from mmdet.apis import inference_detector, init_detector
    det_model = init_detector(cfg.det_config, cfg.det_checkpoint, device=device)
    det_results = []
    for frame in frame_paths:
        result = inference_detector(det_model, frame)
        bboxes = result.pred_instances.bboxes.cpu().numpy()
        scores = result.pred_instances.scores.cpu().numpy()
        valid = scores > cfg.det_score_thr
        det_results.append(bboxes[valid])

    # 步骤3: 姿态估计
    pose_results, _ = pose_inference(
        cfg.pose_config,
        cfg.pose_checkpoint,
        frame_paths,
        det_results,
        device=device
    )

    # 关键点处理逻辑
    max_persons = 2
    num_frames = len(pose_results)

    # 初始化存储容器
    keypoint_list = [[] for _ in range(max_persons)]
    keypoint_score_list = [[] for _ in range(max_persons)]

    for frame_idx, frame_poses in enumerate(pose_results):
        # 数据格式验证
        if not isinstance(frame_poses, list):
            print(f"第{frame_idx}帧姿态数据格式错误，已重置为空列表")
            frame_poses = []

        # 过滤无效数据
        valid_poses = []
        for pose in frame_poses:
            if validate_pose_data(pose, frame_idx):
                valid_poses.append(pose)

        # 按关键点总分排序
        try:
            sorted_persons = sorted(
                valid_poses,
                key=lambda x: np.sum(x['keypoint_scores']),
                reverse=True
            )[:max_persons]
        except Exception as e:
            print(f"第{frame_idx}帧排序异常：{str(e)}，已使用空数据")
            sorted_persons = []

        # 填充关键点数据
        for person_idx in range(max_persons):
            if person_idx < len(sorted_persons):
                # 提取xy坐标并保留两位小数
                kpts = sorted_persons[person_idx]['keypoints'][:, :2].round(2)
                scores = sorted_persons[person_idx]['keypoint_scores'].round(2)
            else:
                # 填充空数据
                kpts = np.zeros((17, 2), dtype=np.float32)
                scores = np.zeros(17, dtype=np.float32)

            keypoint_list[person_idx].append(kpts)
            keypoint_score_list[person_idx].append(scores)

    # 转换为numpy数组并调整维度
    keypoint_array = np.stack([
        np.array(keypoint_list[0]),
        np.array(keypoint_list[1])
    ], axis=0)  # 形状 (2, num_frames, 17, 2)

    keypoint_score_array = np.stack([
        np.array(keypoint_score_list[0]),
        np.array(keypoint_score_list[1])
    ], axis=0)  # 形状 (2, num_frames, 17)

    return {
        'keypoint': keypoint_array,
        'keypoint_score': keypoint_score_array,
        'frame_dir': osp.splitext(osp.basename(video_path))[0],
        'img_shape': (height, width),
        'original_shape': (height, width),
        'total_frames': num_frames,
        'label': label
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True, help='输入视频路径')
    parser.add_argument('--output', required=True, help='输出pkl路径')
    parser.add_argument('--label', type=int, required=True, help='类别标签')
    args = parser.parse_args()

    data = ntu_pose_extraction(args.video, args.label)
    mmengine.dump(data, args.output)
    print(f'姿态数据已保存至: {args.output}')