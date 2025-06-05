# 加载一些辅助函数
import pickle
import os
import shutil
import copy

# 加载 numpy 和 plt
import numpy as np
import matplotlib.pyplot as plt
from math import ceil

# 加载 pytorch
import torch
import torch.nn as nn

# 加载模型
from VAE import TransformerVAE

# 加载 pytorch 优化器
import torch.optim as optim

# 加载 pytorch 的数据集和数据集加载器
from torch.utils.data import Dataset, DataLoader

# 加载分割训练集和测试集的函数库
from torch.utils.data import random_split

# 学习率预热与调度器
from torch.optim.lr_scheduler import OneCycleLR

# 检查 CUDA 是否可用
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Using GPU1:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU instead.")

#################################################################################
## 定义超参数，读取文件

# 是否读取已经保存的权重
Read_Weight = False

# 定义 dropout_rate
DR = 0.1

# 学习率
LR = 0.0001

# 定义训练的总 epoch 数：
num_epochs = 500

# 训练集占比
Train_Rate = 0.85

# 定义批大小
BS = 8

#读取保存 RNA 嵌入 tensor 的字典
# embs_path = '/home/meixy23/TVAE/NEW/output_emb4.pkl'
# with open(embs_path, 'rb') as file:
#     RNA_emb_dict = pickle.load(file)

# # 读取保存标签 tensor 的字典
# labels_path = '/home/meixy23/TVAE/NEW/output_pairs4.pkl'
# with open(labels_path, 'rb') as file:
#     RNA_pair_labels_dict = pickle.load(file)

embs_path = '/home/meixy23/TVAE/Rf1_RNA_emb_dict.pkl'
with open(embs_path, 'rb') as file:
    RNA_emb_dict = pickle.load(file)

# 读取保存标签 tensor 的字典
labels_path = '/home/meixy23/TVAE/Rf1_RNA_pair_labels_dict.pkl'
with open(labels_path, 'rb') as file:
    RNA_pair_labels_dict = pickle.load(file)

#################################################################################
## 定义函数：计算评价指标
def Cal_eval_score(model, dataset, Name):
    # 评估模型
    model.eval()

    # 初始化 TP、FP、FN
    TP, FP, FN = 0, 0, 0

    for idx, (RNA_emb, label) in enumerate(dataset):
        # 把 RNA_emb 和 label 移动到设备上
        # RNA_emb = torch.from_numpy(RNA_emb).to(device)
        # label = torch.from_numpy(label).to(device) if isinstance(label, np.ndarray) else label  # 确保 label 是 tensor
        RNA_emb = RNA_emb.to(device)

        # 得到配对矩阵
        with torch.no_grad():
            out, _, _ = model(RNA_emb.unsqueeze(0))

        # 得到标签和配对矩阵的 numpy 形式
        n_data = (torch.softmax(out[0, :, :], dim=-1)[:, :, 1] > 0.9).to(torch.int16).detach().cpu().numpy()
        pair_mat = label.detach().cpu().numpy()  # 确保 label 是 tensor

        if idx < 5:
            plt.imshow(n_data)
            plt.colorbar()
            plt.savefig(f'./Fig4/{str(idx).zfill(2)}{Name}_预测.png')
            plt.close()
            plt.imshow(pair_mat)
            plt.colorbar()
            plt.savefig(f'./Fig4/{str(idx).zfill(2)}{Name}_真实.png')
            plt.close()

        # 计算TP, FP, FN
        TP += np.sum((n_data == 1) & (pair_mat == 1))
        FP += np.sum((n_data == 1) & (pair_mat == 0))
        FN += np.sum((n_data == 0) & (pair_mat == 1))

    # 计算 Precision 和 Recall
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    # 计算 F1 Score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("准确率:", precision)
    print("召回率:", recall)
    print("F1 分数:", f1_score)

    return f1_score



#################################################################################
## 自定义 RNA 数据集类
class RNADataset(Dataset):
    def __init__(self, data_list):
        """
        data_list 是一个列表，包含 (input_tensor, label_tensor) 元组。
        """
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


# 创建 data_ls
data_ls = []
sample_count = 0  # 控制打印数量
error_count = 0
error_keys = []

for key in RNA_pair_labels_dict.keys():
    key_upper = key.upper()

    if key_upper not in RNA_emb_dict:
        print(f"❌ 找不到嵌入：{key_upper}")
        continue

    # 判断家族，是否为训练集中应该存在的家族
    # Rfam = RNA_pair_labels_dict[key]['rfam_id']
    Rfam = RNA_pair_labels_dict[key][0]
    if Rfam + 1 == 5:
        continue

    # 获取嵌入、家族、配对标签
    Emb = RNA_emb_dict[key_upper]
    # Pair_label = RNA_pair_labels_dict[key]['pair_matrix']
    Pair_label = RNA_pair_labels_dict[key][1]

    if Emb.shape[0] != Pair_label.shape[0]:
        # print(f"❌ 长度不一致！key={key[:30]}..., emb_len={Emb.shape[0]}, label_len={Pair_label.shape[0]}")
        error_count += 1
        error_keys.append(key)
        continue

    # ✅ 打印调试信息（只打印前3个）
    # if sample_count < 3:
    #     print(f"\n✅ Sample {sample_count + 1}")
    #     print(f"Key (原始): {key}")
    #     print(f"Key (大写): {key_upper}")
    #     print(f"Rfam: {Rfam}")
    #     print(f"Embedding shape: {Emb.shape}")
    #     print(f"Pair_label shape: {Pair_label.shape}")
    #     print(f"Embedding type: {type(Emb)}")
    #     print(f"Pair_label type: {type(Pair_label)}")
    #     sample_count += 1

    # 添加到数据列表中
    temp = (Emb, Pair_label)
    data_ls.append(temp)
print(f"\n⚠️ 一共发现 {error_count} 个长度不一致的样本")
# 创建 dataset
dataset = RNADataset(data_ls)


print(f'\n📦 数据集创建成功，总样本数：{len(dataset)}')

# assert False

#################################################################################
## 定义函数：产生方阵中的正负样本掩码（在 01 方阵中，随机生成与 1 数量相等的掩码）
def Get_label_mask(label_tensor):
    lt = copy.deepcopy(label_tensor)

    # 1. 计算张量中值为 1 的数量，这个数量也是方阵中应该添加 2 的个数
    num_ones = (lt == 1).sum()
    num_twos = num_ones.item()

    # 2. 找到所有值为 0 的位置
    zero_indices = torch.nonzero(lt == 0, as_tuple=False)

    # 3. 从值为 0 的位置中随机选择与 1 的数量相同的位置
    perm = torch.randperm(zero_indices.size(0))
    selected_indices = zero_indices[perm[:num_twos]]

    # 4. 将选定位置的值从 0 改为 2
    for idx in selected_indices:
        lt[idx[0], idx[1]] = 2

    # 输出结果
    return lt != 0


#################################################################################
## 定义函数：用来处理 batch 中数据大小不一致的问题
def collate_fn(batch):
    """
    batch 是一个列表，包含 (input_tensor, label_tensor) 元组。
    """

    # 将 batch 列表中 (input_tensor, label_tensor) 元组中的所有 input_tensor 合成一个元组
    # 所有的 label_tensor 合成另一个元组
    inputs, labels = zip(*batch)
    inputs = [torch.from_numpy(inp) if isinstance(inp, np.ndarray) else inp for inp in inputs]

    # 获取每个输入的序列长度
    seq_lengths = [input_tensor.size(0) for input_tensor in inputs]

    # 计算最长的长度
    max_seq_len = max(seq_lengths)

    # 对输入进行填充
    padded_inputs = []
    padding_masks = []
    for input_tensor in inputs:
        pad_size = max_seq_len - input_tensor.size(0)
        if pad_size > 0:
            # 假设 input_tensor 的形状为 [seq_len, feature_dim]
            padding = torch.zeros(pad_size, input_tensor.size(1))
            # 得到掩码
            padding_mask = torch.cat([torch.ones(input_tensor.size(0)), torch.zeros(padding.size(0))], dim=0).to(bool)
            # 得到 padding 后的输入
            padded_input = torch.cat([input_tensor, padding], dim=0)
        else:
            # 得到掩码
            padding_mask = torch.ones(input_tensor.size(0)).to(bool)
            # 得到 padding 后的输入
            padded_input = input_tensor
        # 添加到列表
        padded_inputs.append(padded_input)
        padding_masks.append(~padding_mask)
    # padded_inputs, padding_masks 转化成 tensor：pi, pm
    pi = torch.stack(padded_inputs)
    pm = torch.stack(padding_masks)

    # 对标签进行填充
    padded_labels = []
    label_masks = []
    for label_tensor in labels:
        label_tensor = torch.from_numpy(label_tensor) if isinstance(label_tensor, np.ndarray) else label_tensor

        Len = label_tensor.size(0)
        pad_size = max_seq_len - Len
        # 得到正负样本的掩码
        label_m = Get_label_mask(label_tensor)
        if pad_size > 0:
            # 填充行
            padding_rows = -1 * torch.ones(pad_size, label_tensor.size(1))
            label_tensor = torch.cat([label_tensor, padding_rows], dim=0)
            # 填充列
            padding_cols = -1 * torch.ones(max_seq_len, pad_size)
            label_tensor = torch.cat([label_tensor, padding_cols], dim=1)
        # 得到 padding 后的标签掩码
        label_pm = (label_tensor != -1)
        label_pm[:Len, :Len] = label_m
        # 添加到列表里
        padded_labels.append(label_tensor)
        label_masks.append(label_pm)
    # padded_labels, label_masks 转化成 tensor：pl, lm
    pl = torch.stack(padded_labels)
    lm = torch.stack(label_masks)

    '''print(pl[0, :20, :20])

    plt.imshow(pl[0].to(torch.int16).numpy())
    plt.colorbar()
    plt.savefig('1.png')
    plt.close()
    plt.imshow(lm[0].numpy())
    plt.colorbar()
    plt.savefig('2.png')
    assert False

    print(pl[0][lm[0]].shape)
    print(pl[0][lm[0]][:, 0].sum())
    assert False'''

    # 输出 padding 后的嵌入、嵌入 padding 的位置、padding 的标签、标签 padding 的位置
    return pi, pm, pl, lm


#################################################################################
## 部署模型，创建 DataLoader，定义优化器，定义损失函数

# 对训练集和测试集进行划分（按照超参数中的比例）
total_size = len(dataset)
train_size = int(Train_Rate * total_size)
test_size = total_size - train_size
# 进行划分
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# print(train_dataset[0][1].shape)
# assert False

# 创建 DataLoader（能使用 GPU 的时候，表示在）
if torch.cuda.is_available():
    train_loader = DataLoader(train_dataset, batch_size=BS, collate_fn=collate_fn, shuffle=True, num_workers=32)
    test_loader = DataLoader(test_dataset, batch_size=BS, collate_fn=collate_fn, shuffle=False, num_workers=32)
else:
    train_loader = DataLoader(train_dataset, batch_size=BS, collate_fn=collate_fn, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BS, collate_fn=collate_fn, shuffle=False)

# 部署 KnotFold_Model 模型
model = TransformerVAE(
    input_dim=640,
    hidden_dim=256,
    num_heads=8,
    num_layers=4,
    dropout=DR)
model.to(device)

# 优化器
# optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01, betas=(0.9, 0.999))
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)


# if Read_Weight == True:
#     print('已经读取模型参数')
#     # 读取已经保存的参数
#     model.load_state_dict(torch.load('./Weight/model_checkpoint200.pth', map_location=torch.device(device)))
#     # 加载优化器的状态字典
#     optimizer.load_state_dict(torch.load('optimizer_state.pth'))

# 定义损失函数
criterion = nn.CrossEntropyLoss()

#################################################################################
## 训练模型

# 初始化损失和精度历史记录
train_f1_score_ls = []
test_f1_score_ls = []

# 保存最大的测试集上的 f1 分数
max_test_f1_score = 0
max_test_f1_score_ls = []
import os  # 加在开头

# =================== 在训练前确保文件夹存在 ====================
fig_dir = './Fig4'  # 定义图像保存目录
os.makedirs(fig_dir, exist_ok=True)
kl_max_weight = 0.1
kl_anneal_epochs = 100

epoch = 0
while True:
    epoch += 1

    # 判断是否满足停止条件
    if num_epochs is not None:
        if epoch > num_epochs:
            break

    # 计算 KL 动态权重
    kl_weight = min(kl_max_weight, epoch / kl_anneal_epochs * kl_max_weight)
    # print(kl_weight)

    model.train()
    train_loss = 0.0
    count = 0

    for pi, pm, pl, lm in train_loader:
        # 将数据移动到设备上
        pi = pi.to(device)
        pm = pm.to(device)
        pl = pl.to(device).long()
        lm = lm.to(device)
        attention_mask = pm

        optimizer.zero_grad()
        px, mu, logvar = model(pi, attention_mask)

        ce_loss = criterion(px[lm], pl[lm])
        kl = model.kl_loss(mu, logvar, attention_mask)

        loss = (1 - kl_weight) * ce_loss + kl_weight * kl

        loss.backward()
        optimizer.step()


# 开始训练
# epoch = 0
# while True:
#     epoch += 1
#
#     # 判断是否满足停止条件
#     if num_epochs != None:
#         if epoch > num_epochs:
#             break
#
#     model.train()
#     train_loss = 0.0
#     count = 0
#     # 使用 data_loader 中的数据进行训练
#     for pi, pm, pl, lm in train_loader:
#         # 将数据移动到 GPU 或 CPU 上
#         pi = pi.to(device)
#         pm = pm.to(device)
#         pl = pl.to(device).long()
#         lm = lm.to(device)
#         attention_mask = pm
#         # 梯度清零
#         optimizer.zero_grad()
#         px, mu, logvar = model(pi, attention_mask)
#
#         ce_loss = criterion(px[lm], pl[lm])
#         kl = model.kl_loss(mu, logvar, attention_mask)
#         # if torch.isnan(kl):
#         #     print("KL为NAN")
#         # if torch.isnan(ce_loss):
#         #     raise ValueError("ce_loss 计算后为nan")
#             # print(ce_loss)
#             # print(kl)
#         # assert False
#         loss = 0.8*ce_loss + 0.2*kl
#
#         loss.backward()
#         optimizer.step()

        # # 计算输出
        # out = model(pi, pm)
        # outputs = out[lm]
        #
        # # 计算损失
        # loss = criterion(outputs, pl[lm])
        #
        # # 回传损失
        # loss.backward()
        # optimizer.step()

        # 保存训练损失
        train_loss += loss.item()

    # 求这个 epoch 中的平均损失
    avg_train_loss = train_loss / len(train_loader)

    # 评估测试集
    if epoch % 10 == 0:

        # 打印进度
        print(f'Epoch {epoch + 1}: Train Loss = {avg_train_loss}')

        # 计算评价指标
        print(f'评价训练集（Len：{len(train_dataset)}）效果')
        train_f1_score = Cal_eval_score(model, train_dataset, '训练')
        print(f'评价测试集（Len：{len(test_dataset)}）效果')
        test_f1_score = Cal_eval_score(model, test_dataset, '测试')

        # 添加到列表中
        train_f1_score_ls.append(train_f1_score)
        test_f1_score_ls.append(test_f1_score)

        # 如果当前的测试集上的 f1 分数大于历史最大的 f1 分数，则保存权重
        if max_test_f1_score < test_f1_score:
            # 更新最大的 f1 分数
            max_test_f1_score = test_f1_score
            max_test_f1_score_ls.append(max_test_f1_score)
            # 保存模型参数
            torch.save(model.state_dict(), f'./Weight4/0model_checkpoint.pth')
        else:
            max_test_f1_score_ls.append(max_test_f1_score)

        # 展示当前最大的测试集上的 f1 分数
        print(f'当前测试集最大的 f1 分数是：{max_test_f1_score}')
        print()

        # 画图
        plt.plot(train_f1_score_ls)
        plt.plot(test_f1_score_ls)
        plt.plot(max_test_f1_score_ls)
        plt.title(f'max_test_f1_score: {max_test_f1_score}')
        plt.savefig(os.path.join(fig_dir, f'F1_score_epoch_{epoch}.png'))
        plt.close()

    # 保存模型参数的中间过程
    if epoch % 500 == 0 and epoch != 0:
        torch.save(model.state_dict(), f'./Weight4/model_checkpoint{epoch}.pth')

























