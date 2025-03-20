import os
import re
import numpy as np
import torch
from transformers import BertTokenizer


# 删除html标签
def remove_tags(text):
    re_tag = re.compile(r'<[^>]+>') 
    return re_tag.sub('', text) 

# 读取文件
def read_files(filetype):
    path = r"C:\链时代招新题\你什么意思\aclImdb" # 数据集根目录路径
    file_list = [] # 存储所有文件路径的列表

    # 读取正面评论文件路径
    positive_path = path + filetype +'/pos/'
    for f in os.listdir(positive_path):
        file_list += [positive_path + f] # 将文件的完整路径加入file_list 


    # 读取负面评论文件路径
    negative_path = path + filetype +'/neg/'
    for f in os.listdir(negative_path):
        file_list += [negative_path + f] # 将文件的完整路径加入file_list 

    print('read', filetype, 'files:', len(file_list))

    # 创建标签列表： 正面评论标签设为 1，负面评论标签设为 0
    all_labels = ([1] * 12500 + [0] * 12500)
    all_texts = [] # 用于存储所有评论文本的列表
    
    # 遍历文件路径，读取文件内容
    for fi in file_list:
        with open(fi, encoding='utf8') as file_input:
            all_texts += [remove_tags(" ".join(file_input.readlines()))]
            # 读取所有行并合并成一个字符串，去掉 HTML 标签后存入列表
    return all_labels, all_texts

# 读取训练集和测试集
y_train, x_train = read_files('/train')
y_test, x_test = read_files('/test')

# 将训练集和测试集的标签转换为Numpy数组
y_trian = np.array(y_train)
y_test = np.array(y_test)

# 加载BERT分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# 进行文本编码
def encode_texts(texts,tokenizer,max_length=512):

    encoded = tokenizer(texts, # 输入文本列表
    add_special_tokens=True, # 添加特殊标记 [CLS] 和 [SEP]
    max_length=max_length, # 限制最大长度
    padding='max_length',  # 短文本用 `0` 填充到最大长度
    truncation=True,  # 长文本截断
    return_tensors='np', # 返回 NumPy 数组
    return_attention_mask=True) # 生成注意力掩码

    # 返回文本的token ID和attention mask（用于标识padding部分）
    return encoded['input_ids'],encoded['attention_mask']

# 对训练集和测试集进行编码
x_train_ids,x_train_attention = torch.tensor(encode_texts(x_train,tokenizer))
x_test_ids,x_test_attention = torch.tensor(encode_texts(x_test,tokenizer))

# 将标签转换为tensor
y_train_tensor = torch.tensor(y_train)
y_test_tensor = torch.tensor(y_test)

# 保存训练数据
torch.save(x_train_ids, 'x_train_ids.pt')
torch.save(x_train_attention, 'x_train_attention.pt')
torch.save(y_train_tensor, 'y_train_tensor.pt')

# 保存测试数据
torch.save(x_test_ids, 'x_test_ids.pt')
torch.save(x_test_attention, 'x_test_attention.pt')
torch.save(y_test_tensor, 'y_test_tensor.pt')









