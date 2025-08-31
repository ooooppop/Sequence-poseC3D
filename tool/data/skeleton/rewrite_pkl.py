import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt
import datetime


path = r"D:\WuShuang\mmaction2-main\tools\data\skeleton\predict_rewrite_mnist_10.pkl"
data_list = []
with open(path, 'rb') as f:
    data = pickle.load(f)
    for i in range(len(data['annotations'])):
        # 保持原始切片方式
        original_data = data['annotations'][i]['keypoint'][0]
        sliced_data = original_data[-34:-10, :, :]  # 原始切片

        # 获取当前帧数和需要填充的数量
        current_length = sliced_data.shape[0]
        pad_needed = 24 - current_length

        # 前向填充逻辑
        if pad_needed > 0:
            if current_length == 0:
                raise ValueError(f"样本{i}切片后无数据，请检查原始数据长度")

            # 取切片后的第一帧作为填充模板
            first_frame = sliced_data[0:1]  # 保持维度 (1, 17, 2)
            padding = np.repeat(first_frame, pad_needed, axis=0)

            # 拼接填充数据和原始切片
            processed_data = np.concatenate([padding, sliced_data], axis=0)
        else:
            processed_data = sliced_data[:24]  # 确保不超过24帧

        # 验证最终形状
        assert processed_data.shape == (24, 17, 2), \
            f"样本{i}形状错误: {processed_data.shape}"

        data_list.append(processed_data)

final_array = np.stack(data_list, axis=0)

final_array = final_array.reshape(final_array.shape[0], 24, -1)

# 加载保存的模型
model_save_path = 'seq2seq_model_20250219_225223.pth'

class Seq2Seq(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout_prob=0.3):
        super(Seq2Seq, self).__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob)
        self.decoder = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x, future_steps):
        # 编码器
        encoder_output, (hidden, cell) = self.encoder(x)
        # 解码器
        decoder_input = encoder_output[:, -1, :].unsqueeze(1)
        decoder_outputs = []
        for _ in range(future_steps):
            decoder_output, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
            decoder_input = decoder_output
            decoder_output = self.dropout(decoder_output)
            decoder_outputs.append(self.fc(decoder_output))

        return torch.cat(decoder_outputs, dim=1)


# 确保模型架构与训练时一致
input_size = 34
hidden_size = 256
output_size = 34
num_layers = 3
future_steps = 10

# 实例化模型
model = Seq2Seq(input_size, hidden_size, output_size, num_layers)

# 加载模型权重
model.load_state_dict(torch.load(model_save_path))
model.eval()  # 设置为评估模式

test_output = {}
# 使用测试集进行预测
for j in range(final_array.shape[0]):
    inputs_data = final_array[j, None, :, :]
    inputs_data = torch.tensor(inputs_data, dtype=torch.float32)
    with torch.no_grad():
        test_output[str(j)] = model(inputs_data, future_steps).cpu().numpy()
    print(j)
    temp = test_output[str(j)]
    print(temp)
    data['annotations'][j]['keypoint'][0][-10:, :, :] = temp[0].reshape(10, 17, 2)

with open(r"D:\WuShuang\mmaction2-main\tools\data\skeleton\predict_rewrite_mnist_25.pkl", 'wb') as f:
    pickle.dump(data, f)
