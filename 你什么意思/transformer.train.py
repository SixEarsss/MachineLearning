import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# 加载数据
def load_data():
    x_train_ids = torch.load('x_train_ids.pt')  # 从 x_train_ids.pt 文件加载训练集的 token ID，形状为 (batch_size, seq_len)
    x_train_attention = torch.load('x_train_attention.pt') # 加载训练集的注意力掩码张量，用于标记有效token位置
    y_train_tensor = torch.load('y_train_tensor.pt')  # 读取训练集的标签，返回一维向量(batch, )

    x_test_ids = torch.load('x_test_ids.pt') # 测试集的 token ID，形状 (batch_size, seq_len)
    x_test_attention = torch.load('x_test_attention.pt') # 加载训练集的注意力掩码张量
    y_test_tensor = torch.load('y_test_tensor.pt') # 读取测试集的标签
    
    # 构造TensorDataset，将数据封装成pytorch数据集，便于直接用于Dataloader进行训练
    train_data = TensorDataset(x_train_ids, x_train_attention,y_train_tensor) # train_data：训练集，包含 (x_train_ids,x_train_attention, y_train_tensor)
    test_data = TensorDataset(x_test_ids, x_test_attention,y_test_tensor) # test_data：测试集，包含 (x_test_ids,x_test_attention, y_test_tensor)
    
    return train_data, test_data

train_data, test_data = load_data()

# 创建 DataLoader
batch_size = 16
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Transformer 分类模型
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_heads=4, num_layers=2, num_classes=2):
        # 初始化 Transformer 分类器
        # - vocab_size: 词汇表大小
        # - embed_dim: 词向量的维度（embedding 维度）
        # - num_heads: 多头自注意力的头数
        # - num_layers: Transformer 编码器的层数
        # - num_classes: 分类类别数（默认为二分类任务）
        super().__init__()
        # 词嵌入层：将输入的单词索引转换为 `embed_dim` 维度的向量
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # 定义 Transformer 编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,    # 词向量维度（必须和 embedding 维度一致）
            nhead=num_heads,      # 多头自注意力头数
            dim_feedforward=256   # 前馈神经网络的隐藏层维度（默认 256）
            )

        # 堆叠多个Transformer编码器层
        self.transformer = nn.TransformerEncoder(
            encoder_layer,           # 使用前面预定义的编码器层
            num_layers=num_layers    # 指定Transformer层的数量
            )
        
        # 分类层（全连接层）
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x,attention_mask):
        x = self.embedding(x)  # 将输入形状(batch, seq_len)转化为(batch, seq_len, embed_dim)
        x = x.permute(1, 0, 2)  # Transformer 需要 (seq_len, batch, embed_dim)
        padding_mask = attention_mask == 0  # 转换为布尔型，Padding 的地方是 True
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        x = x.mean(dim=0)  # 取均值，将(seq_len, batch, embed_dim)变为(batch, embed_dim) ，符合全连接层的输入需要
        return self.fc(x)

# 超参数
vocab_size = 30522  # BERT 词表大小（可以改小）
model = TransformerClassifier(vocab_size).to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
optimizer = optim.AdamW(model.parameters(), lr=2e-4)

# 训练模型
def train_model(model, train_loader, criterion, optimizer, epochs=5):
    model.train() # 设定模型为训练模式
    for epoch in range(epochs):
        total_loss = 0   # 用于累加该 epoch 的总损失
        correct = 0      # 用于统计预测正确的样本数
        total = 0        # 用于统计总样本数，以计算准确率
        
        for batch in train_loader:
            optimizer.zero_grad()  # 清空梯度
            input_ids, attention_mask,labels = [x.to(device) for x in batch]
            outputs = model(input_ids,attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            loss.backward() # 反向传播
            optimizer.step() # 更新梯度

            total_loss += loss.item() # .item将tensor转换为普通python类型
            predictions = torch.argmax(outputs, dim=1) # 找到张量中最大值所在的索引，获取预测类别
            correct += (predictions == labels).sum().item()  # 统计预测正确的样本数
            total += labels.size(0)  # 总样本数
        
        print(f"Epoch {epoch+1}: Loss = {total_loss/len(train_loader):.4f}, Accuracy = {correct/total:.4f}")

train_model(model, train_loader, criterion, optimizer)

# 评估模型
def evaluate_model(model, test_loader):
    model.eval() # 设定模型为评估模式
    correct = 0  # 用于统计预测正确的样本数
    total = 0    # 用于统计总样本数，以计算准确率
    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask,labels = [x.to(device) for x in batch]
            outputs = model(input_ids,attention_mask=attention_mask)
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    print(f"Test Accuracy: {correct/total:.4f}")

evaluate_model(model, test_loader)