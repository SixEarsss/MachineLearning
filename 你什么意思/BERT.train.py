import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertForSequenceClassification
from torch.cuda.amp import autocast, GradScaler

# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# 加载数据
def load_data():
    x_train_ids = torch.load('x_train_ids.pt')  # 从 x_train_ids.pt 文件加载训练集的 token ID，形状为 (batch_size, seq_len)
    x_train_attention = torch.load('x_train_attention.pt') # 加载训练集的注意力掩码张量，用于标记有效token位置
    y_train_tensor = torch.load('y_train_tensor.pt')  # 读取训练集的标签，返回一维向量(batch, )
    
    x_test_ids = torch.load('x_test_ids.pt')  # 测试集的 token ID，形状 (batch_size, seq_len)
    x_test_attention = torch.load('x_test_attention.pt') # 加载训练集的注意力掩码张量
    y_test_tensor = torch.load('y_test_tensor.pt')  # 读取测试集的标签
    
    # 构造TensorDataset，将数据封装成pytorch数据集，便于直接用于Dataloader进行训练
    train_data = TensorDataset(x_train_ids, x_train_attention, y_train_tensor)  # train_data：训练集，包含 (x_train_ids,x_train_attention, y_train_tensor)
    test_data = TensorDataset(x_test_ids, x_test_attention, y_test_tensor) # test_data：测试集，包含 (x_test_ids,x_test_attention, y_test_tensor)
     
    return train_data, test_data

train_data, test_data = load_data()

# 创建 DataLoader
batch_size = 16
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# 定义模型
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model.to(device)
for param in model.bert.parameters():
    param.requires_grad = False  # 冻结 BERT 主干网络(将所有参数的梯度设置为不更新)

for param in model.bert.encoder.layer[-4:].parameters():
    param.requires_grad = True  # 只训练最后 4 层


# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=2e-5)

# 混合精度训练
scaler = GradScaler()

def train_model(model, train_loader, criterion, optimizer, epochs=3):
    model.train() # 训练模式
    for epoch in range(epochs):
        # 初始化loss、正确预测数和总样本数
        total_loss = 0
        correct = 0
        total = 0
        for batch in train_loader:
            optimizer.zero_grad()  # 梯度清空
            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            
            # 启用 autocast() 进行混合精度训练
            with autocast():
                outputs = model(input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels)

            scaler.scale(loss).backward() # GradScaler进行损失缩放
            scaler.step(optimizer)  # scaler.step(optimizer) 先反缩放梯度，再更新参数。
            scaler.update()  # 动态调整损失缩放系数，确保 FP16 训练稳定。
            
            total_loss += loss.item()
            predictions = torch.argmax(outputs.logits, dim=1)  # 取 num_labels 维度上的最大值索引，得到最终预测类别
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
        
        print(f"Epoch {epoch+1}: Loss = {total_loss/len(train_loader):.4f}, Accuracy = {correct/total:.4f}")

train_model(model, train_loader, criterion, optimizer)

# 评估模型
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            outputs = model(input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    print(f"Test Accuracy: {correct/total:.4f}")

evaluate_model(model, test_loader)
