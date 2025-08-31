
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt

path = r"D:\WuShuang\mmaction2-main\tools\data\skeleton\mnist_train.pkl"
data_list = []
f = open(path, 'rb')
data = pickle.load(f)
for i in range(len(data['annotations'])):
    data_list.append(data['annotations'][i]['keypoint'][0])

# 确定目标大小
target_size = 34

# 初始化一个空列表来存储调整后的数组
adjusted_data = []

# 遍历每个数组，调整大小
for array in data_list:
    # 如果数组的第一个维度大于等于目标大小，则截取前 target_size 个元素
    rows_to_add = (array.shape[0] // target_size)
    for i in range(rows_to_add):
        adjusted_array = array[target_size * i:target_size * (i+1), :, :]
        adjusted_data.append(adjusted_array)
# print(adjusted_data)
# 将调整后的数组合并成一个形状为 (n, 34, 17, 2) 的数组
final_array = np.stack(adjusted_data, axis=0)

# 打印最终数组的形状
# print(final_array.shape)
split_index = int(len(final_array) * 0.8)
train_data = final_array[:split_index]
test_data = final_array[split_index:]
input_frames = 24
output_frames = 10
# 提取输入和标签
def extract_inputs_and_labels(data, input_frames=24, output_frames=10):
    inputs = data[:, :input_frames, :, :]
    labels = data[:, input_frames:input_frames + output_frames, :, :]
    return inputs, labels

# 应用函数
train_inputs, train_labels = extract_inputs_and_labels(train_data)
test_inputs, test_labels = extract_inputs_and_labels(test_data)

# 调整形状以适应模型
train_inputs = train_inputs.reshape(train_inputs.shape[0], input_frames, -1)
train_labels = train_labels.reshape(train_labels.shape[0], output_frames, -1)
test_inputs = test_inputs.reshape(test_inputs.shape[0], input_frames, -1)
test_labels = test_labels.reshape(test_labels.shape[0], output_frames, -1)


# 定义序列到序列模型
class Seq2Seq(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(Seq2Seq, self).__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, future_steps):
        # 编码器
        encoder_output, (hidden, cell) = self.encoder(x)

        # 解码器
        decoder_input = encoder_output[:, -1, :].unsqueeze(1)  # 使用最后一个时间步的输出作为初始输入
        decoder_outputs = []
        for _ in range(future_steps):
            decoder_output, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
            decoder_input = decoder_output  # 使用当前输出作为下一个时间步的输入
            decoder_outputs.append(self.fc(decoder_output))

        return torch.cat(decoder_outputs, dim=1)


# 参数设置
input_size = 34  # 17个关节点，每个关节点2个坐标
hidden_size = 256
output_size = 34  # 输出也是17个关节点的坐标
num_layers = 3
future_steps = 10  # 预测未来10帧

# 实例化模型
model = Seq2Seq(input_size, hidden_size, output_size, num_layers)

# 损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# 定义学习率调度器
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50000)

# 转换为PyTorch张量
train_inputs = torch.tensor(train_inputs, dtype=torch.float32)
train_labels = torch.tensor(train_labels, dtype=torch.float32)
test_inputs = torch.tensor(test_inputs, dtype=torch.float32)
test_labels = torch.tensor(test_labels, dtype=torch.float32)

# 记录损失
train_losses = []

# 训练模型
num_epochs = 200000
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    output = model(train_inputs, future_steps)
    loss = criterion(output, train_labels)
    loss.backward()
    optimizer.step()

    # 记录损失
    train_losses.append(loss.item())

    # 每隔一定epoch打印一次损失信息
    if (epoch + 1) % 1000 == 0:
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

# 绘制损失曲线
plt.figure(figsize=(12, 6))
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.show()

# 设置模型为评估模式
model.eval()

# 进行预测
with torch.no_grad():
    test_output = model(test_inputs, future_steps)

# 将预测结果转换回numpy数组
test_output_np = test_output.cpu().numpy()

# 打印预测结果
print("Predicted keypoints:")
print(test_output_np)

# 保存预测结果到文件
np.save('predicted_keypoints.npy', test_output_np)

# 如果您想以文本形式保存，可以使用np.savetxt
# 注意：np.savetxt默认保存为文本格式，适合较小的数据集
np.savetxt('predicted_keypoints.txt', test_output_np.reshape(-1, test_output_np.shape[-1]), fmt='%f')

print("Predicted keypoints have been saved to 'predicted_keypoints.npy' and 'predicted_keypoints.txt'.")
