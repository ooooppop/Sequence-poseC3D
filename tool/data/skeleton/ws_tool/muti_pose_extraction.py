# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
import cv2
import numpy as np
import torch
from mmdet.apis import inference_detector, init_detector
from mmpose.apis import inference_topdown, init_model
from mmengine.utils import track_iter_progress
from mmengine.fileio import dump
from tempfile import TemporaryDirectory
# 保持核心功能导入
from mmaction.apis import pose_inference
from mmaction.utils import frame_extract



class MultiPersonTracker:
    """基于IOU的简单多目标跟踪器"""

    def __init__(self, iou_threshold=0.5, max_misses=5):
        self.tracks = []
        self.next_id = 0
        self.iou_thresh = iou_threshold
        self.max_misses = max_misses

    def update(self, detections):
        """更新跟踪状态"""
        # 为每个检测框寻找最佳匹配
        matched = set()
        for track in self.tracks:
            best_iou = 0
            best_idx = -1
            for i, det in enumerate(detections):
                if i in matched:
                    continue
                iou = self.calc_iou(track['bbox'], det)
                if iou > best_iou and iou > self.iou_thresh:
                    best_iou = iou
                    best_idx = i

            if best_idx != -1:
                track['bbox'] = detections[best_idx]
                track['misses'] = 0
                matched.add(best_idx)
            else:
                track['misses'] += 1

        # 移除丢失的目标
        self.tracks = [t for t in self.tracks if t['misses'] < self.max_misses]

        # 添加新目标
        for i, det in enumerate(detections):
            if i not in matched:
                self.tracks.append({
                    'id': self.next_id,
                    'bbox': det,
                    'misses': 0
                })
                self.next_id += 1

        return self.tracks

    @staticmethod
    def calc_iou(box1, box2):
        """计算两个边界框的IOU"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return inter_area / (area1 + area2 - inter_area + 1e-6)


class PoseExtractor:
    """多人姿态提取主类"""

    def __init__(self):
        self.det_config = 'D:/WuShuang/mmaction2-main/demo/demo_configs/faster-rcnn_r50-caffe_fpn_ms-1x_coco-person.py'
        self.det_checkpoint = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco-person/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth'
        self.det_score_thr = 0.5

        # 姿态估计模型配置
        self.pose_config = 'D:/WuShuang/mmaction2-main/demo/demo_configs/td-hm_hrnet-w32_8xb64-210e_coco-256x192_infer.py'
        self.pose_checkpoint = 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth'

        # 初始化模型
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.det_model = init_model(self.det_config, self.det_checkpoint, self.device)
        self.pose_model = init_model(self.pose_config, self.pose_checkpoint, self.device)

    def process_video(self, video_path, label):
        """处理视频主流程"""
        # 视频抽帧
        with TemporaryDirectory() as tmp_dir:
            frame_paths, _ = frame_extract(video_path, out_dir=tmp_dir)

            # 逐帧处理
            tracker = MultiPersonTracker()
            pose_data = []

            for frame_idx, frame_path in track_iter_progress(enumerate(frame_paths)):
                # 目标检测
                det_result = inference_detector(self.det_model, frame_path)
                person_boxes = self._filter_detections(det_result)

                # 目标跟踪
                tracks = tracker.update(person_boxes)

                # 姿态估计
                frame_poses = []
                for track in tracks:
                    pose = inference_topdown(
                        self.pose_model,
                        frame_path,
                        track['bbox'][None, :4],  # 输入需要是二维数组
                        bbox_format='xyxy'
                    )
                    frame_poses.append({
                        'id': track['id'],
                        'keypoints': pose.pred_instances.keypoints[0].cpu().numpy(),
                        'scores': pose.pred_instances.keypoint_scores[0].cpu().numpy()
                    })

                # 按ID排序保证一致性
                pose_data.append(sorted(frame_poses, key=lambda x: x['id']))

            # 格式转换
            return self._format_output(pose_data, video_path, label, frame_paths)

    def _filter_detections(self, det_result):
        """过滤并格式化检测结果"""
        person_boxes = []
        for det in det_result.pred_instances:
            if det.labels.item() == 0 and det.scores.item() > self.det_score_thr:
                box = det.bboxes.cpu().numpy().astype(int)
                person_boxes.append(box)
        return person_boxes

    def _format_output(self, pose_data, video_path, label, frame_paths):
        """转换为MMAaction标准格式"""
        # 获取视频尺寸
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # 确定最大人数
        max_people = max(len(frame) for frame in pose_data)

        # 初始化数据结构
        keypoints = np.zeros((max_people, len(frame_paths), 17, 2), dtype=np.float32)
        scores = np.zeros((max_people, len(frame_paths), 17), dtype=np.float32)

        # 填充数据
        for frame_idx, frame_poses in enumerate(pose_data):
            for person_idx, person in enumerate(frame_poses):
                if person_idx >= max_people:
                    break
                keypoints[person_idx, frame_idx] = person['keypoints'][:, :2]
                scores[person_idx, frame_idx] = person['scores']

        return {
            'keypoint': keypoints,
            'keypoint_score': scores,
            'frame_dir': osp.splitext(osp.basename(video_path))[0],
            'img_shape': (height, width),
            'original_shape': (height, width),
            'total_frames': len(frame_paths),
            'label': label
        }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True, help='输入视频路径')
    parser.add_argument('--output', required=True, help='输出pkl路径')
    parser.add_argument('--label', type=int, required=True, help='类别标签')

    args = parser.parse_args()

    extractor = PoseExtractor()
    result = extractor.process_video(args.video, args.label)
    dump(result, args.output)
    print(f"处理完成！结果保存至 {args.output}")