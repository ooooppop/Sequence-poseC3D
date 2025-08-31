import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt
import datetime
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

path = r"D:\WuShuang\mmaction2-main\tools\data\skeleton\mnist_train.pkl"
data_list = []
with open(path, 'rb') as f:
    data = pickle.load(f)
    for i in range(len(data['annotations'])):
        data_list.append(data['annotations'][i]['keypoint'][0])
# 确定目标大小
target_size = 34

# 调整数据大小
adjusted_data = []
for array in data_list:
    rows_to_add = (array.shape[0] // target_size)
    for i in range(rows_to_add):
        adjusted_array = array[target_size * i:target_size * (i + 1), :, :]
        adjusted_data.append(adjusted_array)

# 将调整后的数组合并成目标形状
final_array = np.stack(adjusted_data, axis=0)

# 数据划分
split_index = int(len(final_array) * 0.8)
train_data = final_array[:split_index]
test_data = final_array[split_index:]

# 提取输入和标签
input_frames = 24
output_frames = 10


def extract_inputs_and_labels(data, input_frames=24, output_frames=10):
    inputs = data[:, :input_frames, :, :]
    labels = data[:, input_frames:input_frames + output_frames, :, :]
    return inputs, labels


train_inputs, train_labels = extract_inputs_and_labels(train_data)
test_inputs, test_labels = extract_inputs_and_labels(test_data)

# 数据预处理与形状调整
train_inputs = train_inputs.reshape(train_inputs.shape[0], input_frames, -1)
train_labels = train_labels.reshape(train_labels.shape[0], output_frames, -1)
test_inputs = test_inputs.reshape(test_inputs.shape[0], input_frames, -1)
test_labels = test_labels.reshape(test_labels.shape[0], output_frames, -1)
train_inputs_1 = train_inputs[0, None, :, :]
train_inputs_1 = torch.tensor(train_inputs_1, dtype=torch.float32)

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

# 使用测试集进行预测
with torch.no_grad():
    test_output = model(train_inputs_1, future_steps)

# 将预测结果转换为 numpy 数组
test_output_np = test_output.cpu().numpy()

# 打印预测结果
print("Predicted keypoints:")
print(test_output_np)

# 保存预测结果到文件
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
npy_filename = f'predicted_keypoints_{timestamp}.npy'
txt_filename = f'predicted_keypoints_{timestamp}.txt'

# 保存为 .npy 文件
np.save(npy_filename, test_output_np)

# 保存为 .txt 文件
np.savetxt(txt_filename, test_output_np.reshape(-1, test_output_np.shape[-1]), fmt='%f')

print(f"Predicted keypoints have been saved to '{npy_filename}' and '{txt_filename}'.")
