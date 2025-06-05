
# 查看PKL文件
# import pickle
#
# # 替换为你实际的文件路径
# file_path = '/home/meixy23/TVAE/Rf1_RNA_pair_labels_dict.pkl'
#
# with open(file_path, 'rb') as f:
#     data = pickle.load(f)
#
# # 打印类型
# print("数据类型：", type(data))
#
# # 如果是字典，打印一个键的内容，并统计RNA序列的总数
# if isinstance(data, dict):
#     total_sequences = len(data)  # 计算字典中键的数量，也就是RNA序列的总数
#     print(f"RNA序列的总数：{total_sequences}")
#     first_key = next(iter(data))  # 获取第一个键
#     print(f"键: {first_key}, 内容: {data[first_key]}")
# elif isinstance(data, list):
#     print(f"列表长度：{len(data)}, 第一个元素内容：{data[0]}")
# else:
#     print("内容：", data[:1])


# 查看配对矩阵的图像

# import torch
# import matplotlib.pyplot as plt
# import os
# import pickle
#
#
# def load_pair_matrices_from_pkl(pkl_file):
#     """
#     从PKL文件加载配对矩阵
#     """
#     with open(pkl_file, 'rb') as f:
#         pair_dict = pickle.load(f)
#     return pair_dict
#
#
# def save_pair_matrices_as_images(pair_dict, output_dir):
#     """
#     将配对矩阵保存为图像，并保存到指定的目录
#     """
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#
#     for seq, (family, pair_matrix) in pair_dict.items():
#         # 创建图像
#         plt.figure(figsize=(8, 8))
#
#         # 绘制热图
#         plt.imshow(pair_matrix.numpy(), cmap='Blues', interpolation='nearest')
#
#         # 设置标题
#         plt.title(f'Pair Matrix for {seq[:10]}...')  # 仅显示序列的前10个字符
#         plt.colorbar()
#
#         # 保存图像
#         img_filename = f'{seq[:10]}_pair_matrix.png'  # 文件名使用序列的前10个字符
#         img_path = os.path.join(output_dir, img_filename)
#         plt.savefig(img_path)
#         plt.close()  # 关闭当前图像
#
#     print(f"配对矩阵图像已保存到 {output_dir}")
#
#
# # 示例用法
# pkl_file = '/home/meixy23/TVAE/NEW/output_pairs1.pkl'  # 你的PKL文件路径
# pair_dict = load_pair_matrices_from_pkl(pkl_file)
#
# output_dir = '/home/meixy23/TVAE/NEW/picture1'  # 图像保存的文件夹
# save_pair_matrices_as_images(pair_dict, output_dir)

# 合成PKL文件（嵌入）

# import os
# import numpy as np
# import torch
# from Bio import SeqIO
# import pickle
#
# # 路径配置
# fasta_folder = '/home/meixy23/VAECNN-transformer/outFiles512quchong/train'
# embedding_folder = '/home/meixy23/VAECNN-transformer/qianru/train'
#
# # 初始化结果字典
# RNA_emb_dict = {}
#
# # 遍历 fasta 文件夹
# for fasta_file in os.listdir(fasta_folder):
#     if not fasta_file.endswith(".fasta"):
#         continue
#
#     fasta_path = os.path.join(fasta_folder, fasta_file)
#     npy_path = os.path.join(embedding_folder, fasta_file + '.npy')  # 例如 seq_1.fasta.npy
#
#     if not os.path.exists(npy_path):
#         print(f"❌ 缺失嵌入文件：{npy_path}，跳过")
#         continue
#
#     # 读取嵌入
#     embedding = np.load(npy_path)  # shape: (L+2, 640)
#     # print(embedding.shape)
#     embedding = embedding.squeeze(0)         # 如果是 (1, L+2, 640) -> (L+2, 640)
#     embedding = embedding[1:-1, :]           # 去掉首尾 token，得到 L×640
#     # print(embedding.shape)
#     # assert False
#     embedding_tensor = torch.tensor(embedding, dtype=torch.float32)
#
#     # 读取序列（统一转大写）
#     records = list(SeqIO.parse(fasta_path, 'fasta'))
#     if len(records) != 1:
#         print(f"⚠️ 文件 {fasta_file} 中存在多条序列，仅处理第一条")
#
#     seq = str(records[0].seq).upper()  # ✅ 强制转大写！
#
#     if len(seq) != embedding_tensor.shape[0]:
#         print(f"❌ 序列长度与嵌入不一致：{fasta_file}，跳过")
#         continue
#
#     # ✅ 使用大写 RNA 序列作为 key
#     RNA_emb_dict[seq] = embedding_tensor
#
# # 保存为 PKL 文件
# with open('rna_dataset.pkl', 'wb') as f:
#     pickle.dump(RNA_emb_dict, f)
#
# print(f"✅ RNA-FM 嵌入已保存，共计 {len(RNA_emb_dict)} 条")



# 合并PKL（配对矩阵）
# import os
# import torch
# import pickle
# from Bio import SeqIO
#
# # 加载FASTA文件，返回 {序列ID: 序列字符串}
# def load_fasta_sequences(fasta_folder):
#     sequences = {}
#     for filename in sorted(os.listdir(fasta_folder)):
#         if filename.endswith(".fasta"):
#             file_path = os.path.join(fasta_folder, filename)
#             for record in SeqIO.parse(file_path, "fasta"):
#                 seq_str = str(record.seq)
#                 seq_id = os.path.splitext(filename)[0]  # 文件名（去掉 .fasta）
#                 sequences[seq_id] = seq_str
#     return sequences
#
# # 加载 PT 文件，返回 {序列ID: 配对矩阵}
# def load_pairing_matrices(pt_folder):
#     matrices = {}
#     for filename in sorted(os.listdir(pt_folder)):
#         if filename.endswith(".pt"):
#             file_path = os.path.join(pt_folder, filename)
#             matrix = torch.load(file_path).float()
#             seq_id = os.path.splitext(filename)[0]  # 文件名（去掉 .pt）
#             matrices[seq_id] = matrix
#     return matrices
#
# # 路径配置
# fasta_folder = '/home/meixy23/VAECNN-transformer/outFiles512quchong/train'
# pt_folder = '/home/meixy23/KnotFold-master/contact/train'
#
# # 加载数据
# sequences = load_fasta_sequences(fasta_folder)
# pairing_matrices = load_pairing_matrices(pt_folder)
#
# # 构建最终 dict：以 RNA 序列字符串为 key，value = (0, 配对矩阵)
# RNA_pair_labels_dict = {}
# for key in pairing_matrices:
#     if key in sequences:
#         rna_seq = sequences[key]
#         matrix = pairing_matrices[key]
#         RNA_pair_labels_dict[rna_seq] = (0, matrix)  # 家族编号设为 0
#
# # 保存为 PKL 文件
# pkl_filename = 'output_data.pkl'
# with open(pkl_filename, 'wb') as f:
#     pickle.dump(RNA_pair_labels_dict, f)
#
# print(f"✅ 已保存为 dict 格式 PKL，key 为 RNA 序列字符串，共计 {len(RNA_pair_labels_dict)} 条记录")

# 查看PKL
# import pickle
#
# # 加载原始PKL文件
# with open('rna_dataset.pkl', 'rb') as f:
#     rna_dict = pickle.load(f)
#
# # 创建一个新的字典，更新key为大写
# rna_dict_cleaned = {key.strip().upper(): value for key, value in rna_dict.items()}
#
# # 保存修改后的字典到新的PKL文件
# with open('rna_dataset.pkl', 'wb') as f:
#     pickle.dump(rna_dict_cleaned, f)
#
# print("RNA序列已转化为大写并保存到新的PKL文件中！")

# 生成配对矩阵
#
# import os
# import pickle
# import numpy as np
#
# def read_fasta(fasta_path):
#     with open(fasta_path, 'r') as f:
#         lines = f.readlines()
#         seq = ''.join([line.strip() for line in lines if not line.startswith('>')])
#         return seq.upper()
#
# def read_ct(ct_path):
#     with open(ct_path, 'r') as f:
#         lines = f.readlines()
#
#     pairs = {}
#     seq = ''
#
#     for line in lines:
#         line = line.strip()
#         if line.startswith("#") or not line:
#             continue
#
#         parts = line.split()
#         if len(parts) < 6:
#             continue
#
#         try:
#             idx = int(parts[0])
#             base = parts[1]
#             pair_idx = int(parts[4])
#         except ValueError:
#             continue
#
#         seq += base
#         if pair_idx != 0:
#             i, j = min(idx, pair_idx) - 1, max(idx, pair_idx) - 1
#             pairs[(i, j)] = 1
#
#     return seq.upper(), pairs
#
# def generate_pair_matrix(length, pair_dict):
#     matrix = np.zeros((length, length), dtype=np.float32)
#     for (i, j) in pair_dict:
#         matrix[i, j] = 1.0
#         matrix[j, i] = 1.0
#     return matrix
#
# def build_pair_and_embedding_dict(fasta_dir, ct_dir, embedding_dir, contact_save_path, embedding_save_path):
#     pair_dict = {}
#     embedding_dict = {}
#     mismatch_count = 0
#     total_count = 0
#
#     for file in os.listdir(fasta_dir):
#         if not file.endswith('.fasta') and not file.endswith('.fa'):
#             continue
#
#         key = os.path.splitext(file)[0]
#         fasta_path = os.path.join(fasta_dir, file)
#         ct_path = os.path.join(ct_dir, key + '.ct')
#         embedding_path = os.path.join(embedding_dir, key + '.fasta.npy')
#
#         if not os.path.exists(ct_path):
#             print(f"❌ 缺失 CT 文件: {key}.ct")
#             continue
#
#         if not os.path.exists(embedding_path):
#             print(f"❌ 缺失嵌入文件: {key}.fasta.npy")
#             continue
#
#         seq_fasta = read_fasta(fasta_path)
#         seq_ct, pair_info = read_ct(ct_path)
#
#         if len(seq_fasta) != len(seq_ct) or seq_fasta != seq_ct:
#             print(f"❌ 序列不一致！{key} 长度: fasta={len(seq_fasta)}, ct={len(seq_ct)}")
#             mismatch_count += 1
#             continue
#
#         embedding = np.load(embedding_path)
#         if embedding.ndim != 3 or embedding.shape[0] != 1 or embedding.shape[2] != 640:
#             print(f"⚠️ 嵌入格式错误: {key}.fasta.npy, 形状为 {embedding.shape}")
#             mismatch_count += 1
#             continue
#
#         embedding = embedding[0, 1:-1, :]
#         if embedding.shape[0] != len(seq_fasta):
#             print(f"❌ 嵌入长度与序列不符: {key}, 嵌入长度 {embedding.shape[0]}, 序列长度 {len(seq_fasta)}")
#             mismatch_count += 1
#             continue
#
#         pair_matrix = generate_pair_matrix(len(seq_fasta), pair_info)
#         pair_dict[str(seq_fasta)] = {
#             "rfam_id": 0,
#             "pair_matrix": pair_matrix
#         }
#         embedding_dict[str(seq_fasta)] = embedding
#         total_count += 1
#
#     print(f"\n✅ 生成完成：匹配成功 {total_count} 条，跳过 {mismatch_count} 条不一致样本。")
#
#     with open(contact_save_path, 'wb') as f:
#         pickle.dump(pair_dict, f)
#     print(f"📦 已保存配对矩阵至：{contact_save_path}")
#
#     with open(embedding_save_path, 'wb') as f:
#         pickle.dump(embedding_dict, f)
#     print(f"📦 已保存嵌入文件至：{embedding_save_path}")
#
# # 示例调用（路径根据实际情况修改）
# build_pair_and_embedding_dict(
#     fasta_dir='/home/meixy23/VAECNN-transformer/outFiles512quchong/train',
#     ct_dir='/home/meixy23/KnotFold-master/CTtrain',
#     embedding_dir='/home/meixy23/VAECNN-transformer/qianru/train',
#     contact_save_path='/home/meixy23/TVAE/contact/contact5.pkl',
#     embedding_save_path='/home/meixy23/TVAE/contact/embedding5.pkl'
# )

#  生成嵌入文件
# import pickle
# import os
# import torch
#
#
# import torch
#
# def read_ct_file(ct_path, seq):
#     length = len(seq)
#     pair_matrix = torch.zeros((length, length))
#
#     with open(ct_path, 'r') as f:
#         lines = f.readlines()
#
#     for line in lines:
#         line = line.strip()
#         if not line:
#             continue
#         # 跳过注释行或包含Energy的行
#         if line.startswith('#') or 'Energy' in line or not line[0].isdigit():
#             continue
#         items = line.split()
#         if len(items) < 5:
#             continue
#         try:
#             idx = int(items[0]) - 1  # 从0开始
#             paired_idx = int(items[4]) - 1  # 也是从0开始
#             if 0 <= paired_idx < length:
#                 pair_matrix[idx, paired_idx] = 1
#                 pair_matrix[paired_idx, idx] = 1
#         except ValueError:
#             continue  # 如果某一行有问题，跳过
#
#     return pair_matrix
#
#
#
# def process_data(input_pkl_path, ct_folder_path, output_feature_pkl_path, output_pair_pkl_path):
#     # 加载原始输入数据
#     with open(input_pkl_path, 'rb') as f:
#         data = pickle.load(f)
#
#     feature_dict = {}
#     pair_dict = {}
#
#     for seq, value in data.items():
#         # 解析value
#         if isinstance(value, tuple) and len(value) == 3:
#             feature_tensor, family_id, seq_name = value
#         else:
#             raise ValueError(f"Unexpected value format for sequence {seq}")
#
#         # 保存第一个pkl（只保留特征）
#         feature_dict[seq] = feature_tensor
#
#         # 生成CT文件路径，假设ct文件名是seq_name+".ct"
#         ct_path = os.path.join(ct_folder_path, f"{seq_name}.ct")
#
#         if not os.path.exists(ct_path):
#             print(f"Warning: CT file not found for {seq_name}")
#             continue
#
#         pair_matrix = read_ct_file(ct_path, seq)
#         if pair_matrix is None:
#             print(f"Warning: CT file does not match sequence for {seq_name}")
#             continue
#
#         # 保存第二个pkl（家族ID和配对矩阵）
#         pair_dict[seq] = (family_id, pair_matrix)
#
#     # 保存两个pkl文件
#     with open(output_feature_pkl_path, 'wb') as f:
#         pickle.dump(feature_dict, f)
#
#     with open(output_pair_pkl_path, 'wb') as f:
#         pickle.dump(pair_dict, f)
#
#     print(f"Saved feature pkl to {output_feature_pkl_path}")
#     print(f"Saved pair pkl to {output_pair_pkl_path}")
#
#
# # 使用示例
# input_pkl = '/home/meixy23/TVAE/NEW/Bases_emb_dict2.pkl'  # 你的第一种格式的输入pkl
# ct_folder = '/home/meixy23/KnotFold-master/CTtrain'  # 存放ct文件的文件夹
# output_feature_pkl = '/home/meixy23/TVAE/NEW/output_emb2.pkl'  # 只包含嵌入特征
# output_pair_pkl = '/home/meixy23/TVAE/NEW/output_pairs2.pkl'  # 包含家族ID和配对矩阵
#
# process_data(input_pkl, ct_folder, output_feature_pkl, output_pair_pkl)


# 查看cpickle文件

# import pickle
# import numpy as np
# import torch
# from collections import namedtuple
#
# # 定义 UFold 使用的数据结构
# RNA_SS_data = namedtuple('RNA_SS_data', 'seq ss_label length name pairs')
#
# # One-hot 映射字典
# base_dict = {'A': 0, 'U': 1, 'C': 2, 'G': 3}
#
# def seq_to_onehot(seq):
#     onehot = np.zeros((len(seq), 4), dtype=np.float32)
#     for i, base in enumerate(seq):
#         if base in base_dict:
#             onehot[i, base_dict[base]] = 1.0
#     return onehot
#
# def contact_to_pairs(matrix, length):
#     """从矩阵提取 base pair，并强制过滤掉非法索引"""
#     matrix = matrix.numpy() if isinstance(matrix, torch.Tensor) else matrix
#     pairs = []
#     for i in range(matrix.shape[0]):
#         for j in range(i + 1, matrix.shape[1]):
#             if matrix[i][j] > 0.5:
#                 if i < length and j < length:
#                     pairs.append([i, j])
#                     pairs.append([j, i])
#     return pairs
#
# def contact_to_label(matrix, length):
#     labels = np.zeros((length, 3), dtype=np.int64)
#     for i in range(length):
#         pair_found = False
#         for j in range(length):
#             if matrix[i][j] > 0.5:
#                 pair_found = True
#                 if i < j:
#                     labels[i][0] = 1  # 左配对
#                     labels[j][2] = 1  # 右配对
#                 break
#         if not pair_found:
#             labels[i][1] = 1  # 未配对
#     return labels
#
# def convert_dict_to_rnassdata(input_path, output_path):
#     with open(input_path, 'rb') as f:
#         data_dict = pickle.load(f)
#
#     dataset = []
#     skipped = 0
#     for idx, (seq_str, (family_id, matrix)) in enumerate(data_dict.items()):
#         length = len(seq_str)
#         name = f"RNA_{idx}"
#
#         try:
#             matrix = matrix.numpy() if isinstance(matrix, torch.Tensor) else matrix
#             matrix = matrix[:length, :length]  # 裁剪防止越界
#
#             seq = seq_to_onehot(seq_str)
#             ss_label = contact_to_label(matrix, length)
#             pairs = contact_to_pairs(matrix, length)
#
#             # 再次过滤（防止数据污染）
#             for i, j in pairs:
#                 if i >= length or j >= length:
#                     raise ValueError(f"Pair ({i},{j}) out of bounds for length {length}")
#
#             dataset.append(RNA_SS_data(seq=seq, ss_label=ss_label, length=length, name=name, pairs=pairs))
#
#         except Exception as e:
#             print(f"⏭️ Skipping {name}: {e}")
#             skipped += 1
#             continue
#
#     with open(output_path, 'wb') as f:
#         pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
#
#     print(f"\n✅ Conversion complete. Saved to: {output_path}")
#     print(f"✅ Total kept: {len(dataset)} | ⏭️ Total skipped: {skipped}")
#
#
# # ====================
# # ✅ 使用方法：改路径
# # ====================
# # 输入你的 .pkl 路径 和 输出的 .cPickle 文件路径
# input_path = '/home/meixy23/TVAE/Rf1_RNA_pair_labels_dict.pkl'
# output_path = '/home/meixy23/UFold/data/TS2.cPickle'
#
# # 执行转换
# convert_dict_to_rnassdata(input_path, output_path)



# #
# # 用法示例
# convert_dict_to_rnassdata('/home/meixy23/TVAE/Rf1_RNA_pair_labels_dict.pkl', '/home/meixy23/UFold/data/TS2.cPickle')

# 转化为bpseq

# import pickle
# import os
# import torch
#
# def convert_pair_matrix_to_bpseq(seq, pair_matrix):
#     """从配对矩阵中提取bpseq格式"""
#     L = len(seq)
#     pairings = [0] * L  # 初始化配对位置
#
#     # 避免重复记录，只取上三角
#     for i in range(L):
#         for j in range(i+1, L):
#             if pair_matrix[i, j] == 1:
#                 pairings[i] = j + 1  # bpseq是1-based
#                 pairings[j] = i + 1
#
#     bpseq_lines = []
#     for i in range(L):
#         bpseq_lines.append(f"{i+1} {seq[i]} {pairings[i]}")
#     return bpseq_lines
#
# def save_bpseq_file(bpseq_lines, output_path):
#     with open(output_path, 'w') as f:
#         for line in bpseq_lines:
#             f.write(line + '\n')
#
# def pkl_to_bpseq(pkl_path, output_dir):
#     os.makedirs(output_dir, exist_ok=True)
#     with open(pkl_path, 'rb') as f:
#         data = pickle.load(f)
#
#     for idx, (seq, (family_id, pair_matrix)) in enumerate(data.items()):
#         if isinstance(pair_matrix, torch.Tensor):
#             pair_matrix = pair_matrix.numpy()
#         bpseq_lines = convert_pair_matrix_to_bpseq(seq, pair_matrix)
#         # 以序号命名，或以序列前10个字符命名也可
#         filename = f"seq_{idx+1}.bpseq"
#         output_path = os.path.join(output_dir, filename)
#         save_bpseq_file(bpseq_lines, output_path)
#
#     print(f"转换完成，生成了 {len(data)} 个bpseq文件，保存在：{output_dir}")
#
# # 示例用法
# pkl_to_bpseq('/home/meixy23/TVAE/NEW/output_pairs1.pkl', '/home/meixy23/UFold/test1')


# 查看pickle文件
# import pickle
# import collections
# RNA_SS_data = collections.namedtuple('RNA_SS_data',
#     'seq ss_label length name pairs')
#
# file_path = '/home/meixy23/e2efold-master/data/rnastralign_all_600/rnastralign_all_600/test_no_redundant_600.pickle'  # 替换为你的文件路径
#
# with open(file_path, 'rb') as f:
#     data = pickle.load(f)
#
# print("数据类型:", type(data))
#
# # 如果是字典，只打印前3个键值对
# if isinstance(data, dict):
#     for i, (k, v) in enumerate(data.items()):
#         print(f"[{i}] Key: {k} -> Type: {type(v)}")
#         if i >= 2:
#             break
#
# # 如果是列表，只打印前3个元素
# elif isinstance(data, list):
#     for i, item in enumerate(data[:3]):
#         print(f"[{i}] {item} (Type: {type(item)})")
#
# # 其他类型，直接打印
# else:
#     print(data)


# BPSEQ转化为CT文件
# def bpseq_to_ct(bpseq_path, ct_path):
#     """
#     将 BPSEQ 格式文件转换为 CT 格式文件。
#     """
#     bases = []
#     pairings = []
#
#     # 读取 bpseq 文件
#     with open(bpseq_path, 'r') as f:
#         for line in f:
#             if line.startswith('#') or not line.strip():
#                 continue
#             parts = line.strip().split()
#             if len(parts) < 3:
#                 continue
#             idx, base, pair = int(parts[0]), parts[1], int(parts[2])
#             bases.append(base)
#             pairings.append(pair)
#
#     n = len(bases)
#
#     with open(ct_path, 'w') as f:
#         # 写入 header
#         f.write(f"{n}  ENERGY = 0\n")
#         for i in range(n):
#             index = i + 1
#             base = bases[i]
#             prev_idx = i if i > 0 else 0
#             next_idx = i + 2 if i < n - 1 else 0
#             pair = pairings[i]
#             f.write(f"{index} {base} {prev_idx} {next_idx} {pair} {index}\n")
#
#     print(f"已转换: {bpseq_path} → {ct_path}")
#
#
# # ✅ 示例调用：
# bpseq_file = "/home/meixy23/mxfold2-master/output_bpseq/RNA_sequence.bpseq"
# ct_file = "/home/meixy23/mxfold2-master/example2.ct"
# bpseq_to_ct(bpseq_file, ct_file)


# 从CT中提取RNA序列
#
# def extract_sequence_from_ct(ct_path):
#     """
#     从 CT 文件中提取 RNA 序列。
#     """
#     sequence = []
#     with open(ct_path, 'r') as f:
#         lines = f.readlines()[1:]  # 跳过第一行标题
#         for line in lines:
#             parts = line.strip().split()
#             if len(parts) >= 2:
#                 sequence.append(parts[1].upper())
#     return ''.join(sequence)
#
# def save_sequence_to_fasta(sequence, fasta_path, header="RNA_sequence"):
#     """
#     将 RNA 序列保存为 FASTA 文件格式。
#     """
#     with open(fasta_path, 'w') as f:
#         f.write(f">{header}\n")
#         # 可选：每行限制为60字符，标准FASTA风格
#         for i in range(0, len(sequence), 60):
#             f.write(sequence[i:i+60] + "\n")
#
# # 示例路径
# ct_file = "/home/meixy23/mxfold2-master/sequence_18.ct"
# fasta_file = "/home/meixy23/mxfold2-master/output18.fasta"
#
# # 提取并保存
# rna_seq = extract_sequence_from_ct(ct_file)
# save_sequence_to_fasta(rna_seq, fasta_file)
#
# print(f"已将 RNA 序列保存为 {fasta_file}")

# 数据生成完整流程
#
# import os
# import pickle
# import torch
# import fm
# from tqdm import tqdm
#
# # ------------------------------
# # 配置 GPU 设备：使用 GPU 1（编号从 0 开始）
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")
#
# # 配置 RNA-FM 模型
# class Config:
#     class MODEL:
#         BACKBONE_NAME = "rna-fm"
#         PAIRWISE_PREDICTOR_NAME = "none"
#         BACKBONE_FROZEN = 1
#         MODEL_LOCATION = "/home/meixy23/VAECNN-transformer/RNA-FM/RNA-FM_pretrained.pth"  # 修改为你的路径
#
# cfg = Config()
# model = fm.downstream.build_model(cfg).to(device)
# alphabet = model.backbone_alphabet
# model.eval()
# batch_converter = alphabet.get_batch_converter()
# # ------------------------------
#
# # 多个 CT 文件夹路径
# ct_folders = [
#     '/home/meixy23/KnotFold-master/CTtrain',
#     '/home/meixy23/KnotFold-master/CTVAL',
#     '/home/meixy23/KnotFold-master/CTtest'
# ]
#
# # 输出文件路径
# output_feature_pkl = '/home/meixy23/TVAE/NEW/output_emb4.pkl'     # RNA序列 -> 嵌入
# output_pair_pkl = '/home/meixy23/TVAE/NEW/output_pairs4.pkl'     # RNA序列 -> (family_id, pair_matrix)
#
# # ========================
# # 辅助函数：读取 CT 文件生成配对矩阵
# def read_ct_file(ct_path, seq):
#     length = len(seq)
#     pair_matrix = torch.zeros((length, length))
#
#     with open(ct_path, 'r') as f:
#         lines = f.readlines()
#
#     for line in lines:
#         line = line.strip()
#         if not line or line.startswith('#') or 'Energy' in line or not line[0].isdigit():
#             continue
#         items = line.split()
#         if len(items) < 5:
#             continue
#         try:
#             idx = int(items[0]) - 1
#             paired_idx = int(items[4]) - 1
#             if 0 <= idx < length and 0 <= paired_idx < length:
#                 pair_matrix[idx, paired_idx] = 1
#                 pair_matrix[paired_idx, idx] = 1
#         except ValueError:
#             continue
#     return pair_matrix
#
# # ========================
# # 主流程
# feature_dict = {}
# pair_dict = {}
# count = 0
#
# for ct_folder in ct_folders:
#     ct_files = [f for f in os.listdir(ct_folder) if f.endswith('.ct')]
#
#     for ct_file in tqdm(ct_files, desc=f"Processing folder: {ct_folder}"):
#         key = ct_file.replace('.ct', '')
#         ct_path = os.path.join(ct_folder, ct_file)
#
#         # 读取 RNA 序列
#         with open(ct_path, 'r') as f:
#             lines = f.readlines()
#         valid_lines = [line for line in lines if not line.startswith('#') and line.strip()]
#         if len(valid_lines) < 2:
#             continue
#         RNA_seq = ''.join([line.split()[1] for line in valid_lines[1:] if len(line.split()) >= 5])
#         if len(RNA_seq) == 0 or len(RNA_seq) >= 511:
#             continue
#
#         # 1. 嵌入计算（GPU）
#         try:
#             data = [("RNA", RNA_seq)]
#             batch_labels, batch_strs, batch_tokens = batch_converter(data)
#             batch_tokens = batch_tokens.to(device)  # 移动到GPU
#             input_data = {
#                 "description": batch_labels,
#                 "token": batch_tokens
#             }
#             with torch.no_grad():
#                 results = model(input_data)
#                 embedding = results['representations'][12][0, :, :][1:-1, :].cpu()  # 从GPU转回CPU保存
#         except Exception as e:
#             print(f"[Error] Embedding failed for {key}: {e}")
#             continue
#
#         # 2. 配对矩阵
#         try:
#             pair_matrix = read_ct_file(ct_path, RNA_seq)
#             if embedding.shape[0] != pair_matrix.shape[0]:
#                 print(f"[Warning] Length mismatch for {key}: emb={embedding.shape[0]}, pair={pair_matrix.shape[0]}")
#                 continue
#         except Exception as e:
#             print(f"[Error] Pair matrix failed for {key}: {e}")
#             continue
#
#         # 3. 保存（确保一一对应）
#         feature_dict[RNA_seq] = embedding
#         pair_dict[RNA_seq] = (0, pair_matrix)  # family_id 设为0，如有需要可替换
#
#         count += 1
#         if count >= 40000:
#             break
#     if count >= 40000:
#         break
#
# # ========================
# # 保存 pkl 文件
# with open(output_feature_pkl, 'wb') as f:
#     pickle.dump(feature_dict, f)
# with open(output_pair_pkl, 'wb') as f:
#     pickle.dump(pair_dict, f)
#
# print(f"\n✅ Finished! Saved {count} RNAs.")
# print(f" - Feature PKL: {output_feature_pkl}")
# print(f" - Pair PKL: {output_pair_pkl}")


# 生成小提琴图


# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# import os
#
# # 设置输出目录
# output_dir = "/home/meixy23/TVAE/MXfold2"
# os.makedirs(output_dir, exist_ok=True)
#
# # 正确读取数据（支持空格或tab等分隔符）
# df_raw = pd.read_csv("/home/meixy23/TVAE/MXfold2/result1.csv", sep=r"\s+", header=None, engine='python')
#
# # 取出 'Name', 'Precision', 'Recall', 'F1'
# df_metrics = df_raw[[0, 8, 9, 10]]
# df_metrics.columns = ['Name', 'Precision', 'Recall', 'F1']
#
# # 转换为长表（便于画图）
# df_melted = pd.melt(df_metrics, id_vars='Name',
#                     value_vars=['Precision', 'Recall', 'F1'],
#                     var_name='Metric', value_name='Score')
#
# # 设置 Seaborn 风格
# sns.set(style="whitegrid")
#
# # 创建横向小提琴图
# plt.figure(figsize=(8, 4))  # 瘦长图
# sns.violinplot(
#     data=df_melted,
#     x='Score',
#     y='Metric',  # 横着的关键在这里
#     palette='Set2',
#     cut=0,
#     inner='box',
#     linewidth=1.2,
#     width=0.6
# )
#
# plt.xlim(0, 1)
# plt.xlabel("Score")
# plt.ylabel("Metric")
# plt.title("Distribution of Precision / Recall / F1")
#
# # 保存图像
# save_path = os.path.join(output_dir, "horizontal_violin.svg")
# plt.tight_layout()
# plt.savefig(save_path, format='svg')
# plt.close()
#
# print(f"✅ 横向小提琴图已保存至: {save_path}")




# 生成表格

# 环境重置后重新执行绘制表格图像的代码
# import matplotlib
# matplotlib.use('Agg')  # ✅ 适用于无GUI服务器环境
# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
# import os
#
# # 定义模型及对应指标
# models = [
#     "TVAE", "UFold", "SPOT-RNA", "RNAstructure", "RNAFold",
#     "MXFold", "MXFold2", "KnotFold", "E2EFold"
# ]
# precision = [0.8280, 0.5681, 0.5417, 0.6380, 0.5324, 0.5829, 0.6101, 0.7164, 0.2697]
# recall    = [0.8928, 0.6687, 0.5987, 0.7487, 0.6487, 0.6434, 0.7096, 0.8327, 0.2574]
# f1_score  = [0.8592, 0.5953, 0.5778, 0.6723, 0.5683, 0.6125, 0.6433, 0.7677, 0.2531]
#
# # 创建 DataFrame
# df = pd.DataFrame({
#     'Model': models,
#     'Precision': precision,
#     'Recall': recall,
#     'F1 Score': f1_score
# })
#
# # 设置图像输出目录
# output_dir = "./violin_plots"
# os.makedirs(output_dir, exist_ok=True)
# output_path = os.path.join(output_dir, "model_comparison_table.png")
#
# # 绘制表格
# fig, ax = plt.subplots(figsize=(10, 3))
# ax.axis('off')
# table = ax.table(
#     cellText=df.round(4).values,
#     colLabels=df.columns,
#     cellLoc='center',
#     loc='center'
# )
# table.auto_set_font_size(False)
# table.set_fontsize(11)
# table.scale(1.2, 1.5)
#
# # 保存为PNG
# plt.savefig(output_path, dpi=300, bbox_inches='tight')
# plt.close()
#
# output_path


# 提取PKL数据（单个）

# import pickle
#
# # === 设置路径 ===
# input_path = '/home/meixy23/TVAE/Rf1_RNA_emb_dict.pkl'         # 原始PKL路径
# output_path = '/home/meixy23/TVAE/predict/sample_index_5.pkl'  # 输出文件路径
# target_index = 5  # 想取第几个（从0开始计数）
#
# # === 加载原始PKL文件 ===
# with open(input_path, 'rb') as f:
#     data_dict = pickle.load(f)
#
# # === 提取第 N 个样本 ===
# if isinstance(data_dict, dict):
#     keys = list(data_dict.keys())
#
#     if target_index < len(keys):
#         key = keys[target_index]
#         value = data_dict[key]
#         one_sample_dict = {key: value}
#
#         # === 保存为新PKL文件 ===
#         with open(output_path, 'wb') as out_f:
#             pickle.dump(one_sample_dict, out_f)
#
#         print(f'✅ 已成功提取第 {target_index} 个样本并保存至: {output_path}')
#         print(f'序列 key: {key}')
#     else:
#         print(f'❌ 输入索引 {target_index} 超出数据范围（共 {len(keys)} 个样本）')
# else:
#     print(f'❌ 加载的对象不是 dict，而是: {type(data_dict)}')


# 提取PKL数据（多个个）
# === 提取第 start_idx 到 end_idx-1 的样本 ===

import pickle

input_path = '/home/meixy23/TVAE/Rf1_RNA_emb_dict.pkl'
output_path = '/home/meixy23/TVAE/predict/sample.pkl'

start_idx = 10
end_idx = 30  # 不包含

with open(input_path, 'rb') as f:
    data_dict = pickle.load(f)

if isinstance(data_dict, dict):
    keys = list(data_dict.keys())
    selected_keys = keys[start_idx:end_idx]
    extracted_dict = {k: data_dict[k] for k in selected_keys}

    with open(output_path, 'wb') as out_f:
        pickle.dump(extracted_dict, out_f)

    print(f'✅ 成功提取 {len(extracted_dict)} 个样本: [{start_idx}, {end_idx})，保存至: {output_path}')
    print(f'示例 key 列表: {[k[:20] for k in selected_keys]}')
else:
    print(f'❌ 加载的对象不是 dict，而是: {type(data_dict)}')


# 查看pkl文件的数量
# import pickle
#
# def count_rna_entries_in_file(pkl_file_path):
#     with open(pkl_file_path, 'rb') as f:
#         data = pickle.load(f)
#         if isinstance(data, dict):
#             entry_count = len(data)
#             print(f"文件中 RNA 序列的数量为: {entry_count}")
#         else:
#             print("错误：该 pkl 文件不是一个字典结构。")
#
# # 用法：替换为你自己的文件路径
# pkl_file_path = '/home/meixy23/TVAE/NEW/output_emb4.pkl'
# count_rna_entries_in_file(pkl_file_path)

# 匹配CT文件

# import os
# import shutil
#
# def read_fasta_seq(fasta_path):
#     seq = []
#     with open(fasta_path, 'r') as f:
#         for line in f:
#             if line.startswith('>'):
#                 continue
#             seq.append(line.strip().upper())
#     return ''.join(seq)
#
#
# import numpy as np
#
#
# def parse_ct_file(filepath):
#     bases = []
#     pairs = []
#
#     with open(filepath, 'r') as f:
#         lines = f.readlines()
#
#     data_lines = [line.strip() for line in lines if line.strip() and not line.startswith('ENERGY')]
#
#     for line in data_lines[1:]:  # 跳过第一行注释
#         parts = line.strip().split()
#         if len(parts) < 6:
#             continue
#         try:
#             idx = int(parts[0])
#             base = parts[1].upper()
#             pair = int(parts[4])
#
#             if base not in ['A', 'U', 'G', 'C', 'T', 'N']:  # 支持 N 和 T
#                 # print(f"⚠️ 警告: CT文件 {filepath} 行中出现非法碱基 '{base}'，跳过该行。")
#                 continue
#
#             bases.append(base)
#             pairs.append(pair)
#         except ValueError:
#             # print(f"⚠️ 警告: 非法行（非整数索引）→ {line}")
#             continue
#
#     L = len(bases)
#     ct_matrix = np.zeros((L, L), dtype=int)
#
#     for i, j in enumerate(pairs):
#         if j > 0 and j - 1 < L:
#             ct_matrix[i][j - 1] = 1
#             ct_matrix[j - 1][i] = 1
#
#     return ''.join(bases), ct_matrix
#
#
# def copy_matching_ct_files_by_seq(fasta_dir, ct_dir, output_dir):
#     os.makedirs(output_dir, exist_ok=True)
#
#     # 读取FASTA序列
#     fasta_seq_dict = {}
#     fasta_count = 0
#     for fasta_file in os.listdir(fasta_dir):
#         if fasta_file.endswith('.fasta') or fasta_file.endswith('.fa'):
#             fasta_path = os.path.join(fasta_dir, fasta_file)
#             seq = read_fasta_seq(fasta_path)
#             fasta_seq_dict[seq] = fasta_file
#             fasta_count += 1
#             # print(f"[FASTA] {fasta_file} → 长度: {len(seq)} 前10个碱基: {seq[:10]}")
#
#     print(f"\n共读取 {fasta_count} 个FASTA文件\n")
#
#     # 读取并匹配CT文件
#     ct_files = [f for f in os.listdir(ct_dir) if f.endswith('.ct')]
#     copied_count = 0
#     ct_count = 0
#
#     for ct_file in ct_files:
#         ct_path = os.path.join(ct_dir, ct_file)
#         seq_ct, _ = parse_ct_file(ct_path)
#         ct_count += 1
#         # print(f"[CT] {ct_file} → 长度: {len(seq_ct)} 前10个碱基: {seq_ct[:10]}")
#
#         if seq_ct in fasta_seq_dict:
#             dst_path = os.path.join(output_dir, ct_file)
#             shutil.copy2(ct_path, dst_path)
#             copied_count += 1
#             # print(f"✅ 匹配成功: {ct_file} → 复制")
#         # else:
#         #     print(f"❌ 无匹配: {ct_file} → 不在FASTA中")
#
#     print(f"\n共处理 {ct_count} 个CT文件")
#     print(f"✅ 总共复制了 {copied_count} 个匹配的CT文件到 {output_dir}")
#
# # 示例调用，替换为你的实际路径
# copy_matching_ct_files_by_seq(
#     fasta_dir='/home/meixy23/VAECNN-transformer/outFiles512quchong/val',
#     ct_dir='/home/meixy23/KnotFold-master/ctFiles',
#     output_dir='/home/meixy23/KnotFold-master/CTVAL'
# )



# 示例调用，替换为你的实际路径
# copy_matching_ct_files(
#     rna_seq_dir='/home/meixy23/VAECNN-transformer/outFiles512quchong/test',
#     ct_dir='/home/meixy23/KnotFold-master/ctFiles',
#     output_dir='/home/meixy23/KnotFold-master/CTtest'
# )


