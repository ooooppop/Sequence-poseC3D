import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt

# 加载数据
path = r"D:\WuShuang\mmaction2-main\tools\data\skeleton\mnist_train.pkl"
data_list = []
with open(path, 'rb') as f:
    data = pickle.load(f)
    for i in range(len(data['annotations'])):
        data_list.append(data['annotations'][i]['keypoint'][0])

# 确定目标大小
target_size = 34

# 调整数组大小
adjusted_data = []
for array in data_list:
    rows_to_add = (array.shape[0] // target_size)
    for i in range(rows_to_add):
        adjusted_array = array[target_size * i:target_size * (i + 1), :, :]
        adjusted_data.append(adjusted_array)

# 将调整后的数组合并成形状为 (n, 34, 17, 2) 的数组
final_array = np.stack(adjusted_data, axis=0)
train_data, test_data = train_test_split(final_array, test_size=0.2, random_state=42)
input_frames, output_frames = 24, 10


# 提取输入和标签
def extract_inputs_and_labels(data, input_frames=24, output_frames=10):
    inputs = data[:, :input_frames, :, :]
    labels = data[:, input_frames:input_frames + output_frames, :, :]
    return inputs, labels


train_inputs, train_labels = extract_inputs_and_labels(train_data)
test_inputs, test_labels = extract_inputs_and_labels(test_data)

# 调整形状为适应模型 (batch_size, seq_len, feature_size)
train_inputs = train_inputs.reshape(train_inputs.shape[0], input_frames, -1)
train_labels = train_labels.reshape(train_labels.shape[0], output_frames, -1)
test_inputs = test_inputs.reshape(test_inputs.shape[0], input_frames, -1)
test_labels = test_labels.reshape(test_labels.shape[0], output_frames, -1)

# 转换为 PyTorch 张量
train_inputs = torch.tensor(train_inputs, dtype=torch.float32)
train_labels = torch.tensor(train_labels, dtype=torch.float32)
test_inputs = torch.tensor(test_inputs, dtype=torch.float32)
test_labels = torch.tensor(test_labels, dtype=torch.float32)


# 定义 Seq2Seq 模型
class Seq2Seq(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(Seq2Seq, self).__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, future_steps, initial_decoder_input=None):
        # 编码器
        encoder_output, (hidden, cell) = self.encoder(x)

        # 使用提供的初始解码器输入作为解码器起点
        if initial_decoder_input is not None:
            decoder_input = initial_decoder_input.unsqueeze(1)  # (batch_size, 1, input_size)
        else:
            decoder_input = encoder_output[:, -1, :].unsqueeze(1)  # 使用编码器最后输出

        decoder_outputs = []
        for _ in range(future_steps):
            decoder_output, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
            decoder_input = decoder_output  # 当前输出作为下一个时间步的输入
            decoder_outputs.append(self.fc(decoder_output))

        return torch.cat(decoder_outputs, dim=1)


# 参数设置
input_size, hidden_size, output_size = 34, 128, 34
num_layers, future_steps = 2, 10

# 实例化模型
model = Seq2Seq(input_size, hidden_size, output_size, num_layers)

# 损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50000)

# 训练参数
num_epochs = 10000
train_losses = []

initial_decoder_input = train_inputs[:, -1, :]  # (batch_size, input_size)

# 扩展为 (batch_size, future_steps, input_size)
initial_decoder_input_step = initial_decoder_input.unsqueeze(1).expand(-1, future_steps, -1)

# 定义训练过程
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass，使用训练集最后一帧扩展的标签作为解码器初始输入
    output = model(train_inputs, future_steps, initial_decoder_input_step)

    # 计算损失
    loss = criterion(output, train_labels)
    loss.backward()
    optimizer.step()

    # 记录损失
    train_losses.append(loss.item())

    # 打印损失信息
    if (epoch + 1) % 100 == 0:
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')