import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt
import datetime

# 读取数据
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

# 数据归一化
mean = train_inputs.mean()
std = train_inputs.std()
train_inputs = (train_inputs - mean) / std
test_inputs = (test_inputs - mean) / std

# 转换为PyTorch张量
train_inputs = torch.tensor(train_inputs, dtype=torch.float32)
train_labels = torch.tensor(train_labels, dtype=torch.float32)
test_inputs = torch.tensor(test_inputs, dtype=torch.float32)
test_labels = torch.tensor(test_labels, dtype=torch.float32)


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
        decoder_input = encoder_output[:, -1, :].unsqueeze(1)
        decoder_outputs = []
        for _ in range(future_steps):
            decoder_output, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
            decoder_input = decoder_output
            decoder_outputs.append(self.fc(decoder_output))

        return torch.cat(decoder_outputs, dim=1)


# 参数设置
input_size = 34
hidden_size = 256
output_size = 34
num_layers = 3
future_steps = 10

# 实例化模型
model = Seq2Seq(input_size, hidden_size, output_size, num_layers)

# 损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50000)

# Batch Loader
batch_size = 64
train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(train_inputs, train_labels),
    batch_size=batch_size, shuffle=True
)

# 记录损失
train_losses = []

# 训练模型
num_epochs = 50000
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_idx, (batch_inputs, batch_labels) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(batch_inputs, future_steps)
        loss = criterion(output, batch_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
        optimizer.step()
        total_loss += loss.item()

        # 打印当前批次进度
        if (batch_idx + 1) % 100 == 0:
            print(f"[Batch {batch_idx + 1}/{len(train_loader)}] Loss: {loss.item():.6f}")

    scheduler.step()  # 调整学习率

    # 记录并打印损失和学习率
    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)

    current_lr = optimizer.param_groups[0]['lr']
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.6f}, Learning Rate: {current_lr:.6f}")

# 可视化训练损失
plt.plot(train_losses)
plt.title('Training Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
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

# 获取当前时间戳，用于文件命名
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

# 保存预测结果到文件，文件名中加入时间戳
npy_filename = f'predicted_keypoints_{timestamp}.npy'
txt_filename = f'predicted_keypoints_{timestamp}.txt'

# 保存为.npy文件
np.save(npy_filename, test_output_np)

# 保存为.txt文件
np.savetxt(txt_filename, test_output_np.reshape(-1, test_output_np.shape[-1]), fmt='%f')

print(f"Predicted keypoints have been saved to '{npy_filename}' and '{txt_filename}'.")