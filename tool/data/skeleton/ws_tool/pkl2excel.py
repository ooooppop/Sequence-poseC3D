import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm


def pkl_to_excel(pkl_path, excel_path):
    # 读取PKL文件
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    annotations = data['annotations']
    rows = []

    # 遍历每个视频标注
    for ann in tqdm(annotations, desc="Processing videos"):
        frame_dir = ann['frame_dir']
        label = ann['label']
        keypoints = ann['keypoint']  # 形状 [M, T, V, C]
        scores = ann.get('keypoint_score', None)  # 形状 [M, T, V]

        M, T, V, C = keypoints.shape

        # 遍历每个人物
        for m in range(M):
            # 遍历每帧
            for t in range(T):
                # 遍历每个关节点
                for v in range(V):
                    coord = keypoints[m, t, v]
                    row = {
                        '视频ID': frame_dir,
                        '动作标签': label,
                        '人物ID': m + 1,
                        '帧序号': t + 1,
                        '关节点ID': v + 1,
                        'X坐标': coord[0] if C >= 1 else np.nan,
                        'Y坐标': coord[1] if C >= 2 else np.nan,
                        'Z坐标': coord[2] if C >= 3 else np.nan,
                        '置信度': scores[m, t, v] if scores is not None else 1.0
                    }
                    rows.append(row)

    # 创建DataFrame
    df = pd.DataFrame(rows)

    # 写入Excel（自动分sheet处理大数据）
    with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
        # 按视频ID分sheet存储
        for video_id in df['视频ID'].unique():
            video_df = df[df['视频ID'] == video_id]

            # 截断过长的sheet名称
            sheet_name = f"{video_id[:20]}_数据" if len(video_id) > 20 else f"{video_id}_数据"

            # 写入数据
            video_df.to_excel(
                writer,
                sheet_name=sheet_name,
                index=False,
                columns=['人物ID', '帧序号', '关节点ID', 'X坐标', 'Y坐标', 'Z坐标', '置信度']
            )

            # 添加汇总表
            summary_df = video_df.groupby(['人物ID', '关节点ID']).agg({
                '置信度': 'mean',
                'X坐标': ['min', 'max'],
                'Y坐标': ['min', 'max']
            })
            summary_df.to_excel(writer, sheet_name=f"{sheet_name}_汇总")


if __name__ == "__main__":
    pkl_path = r"D:\WuShuang\mmaction2-main\tools\data\skeleton\mnist_train.pkl"
    pkl_to_excel(pkl_path, 'output.xlsx')