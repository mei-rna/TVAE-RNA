#   生成小提琴图
import torch
import torch.nn.functional as F
import pickle
import os
import numpy as np
import tempfile
import subprocess
import matplotlib.pyplot as plt
from VAE3 import TransformerVAE

# === 参数设置 ===
model_path = '/home/meixy23/TVAE/Weight2/0model_checkpoint.pth'
input_emb_path = '/home/meixy23/TVAE/Rf1_RNA_emb_dict.pkl'
label_path = '/home/meixy23/TVAE/Rf1_RNA_pair_labels_dict.pkl'
output_pkl_path = '/home/meixy23/TVAE/predict/Rf11.pkl'
violin_plot_path = '/home/meixy23/TVAE/predict/violin_FTVAE111.svg'

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# === 加载模型 ===
model = TransformerVAE(input_dim=640, hidden_dim=256, num_heads=8, num_layers=4, dropout=0.1).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# === 加载数据 ===
with open(input_emb_path, 'rb') as f:
    RNA_emb_dict = pickle.load(f)
with open(label_path, 'rb') as f:
    label_dict = pickle.load(f)


# === 匈牙利算法===
def run_hungarian_algorithm(prob_matrix, lambda_param=4.2, eps=1e-8):
    """
    使用匈牙利算法进行RNA二级结构预测
    """
    n = prob_matrix.shape[0]

    # 生成背景概率矩阵（假设为0.5或其他合理值）
    bg_matrix = np.full_like(prob_matrix, 0.5)

    # 使用临时文件保存矩阵
    with tempfile.TemporaryDirectory() as tmpdir:
        fg_path = os.path.join(tmpdir, "foreground.mat")
        bg_path = os.path.join(tmpdir, "background.mat")

        # 保存矩阵到文件
        np.savetxt(fg_path, prob_matrix, fmt="%.10f", delimiter=" ")
        np.savetxt(bg_path, bg_matrix, fmt="%.10f", delimiter=" ")

        # 调用编译好的匈牙利算法程序
        exe_path = "/home/meixy23/TVAE/hungarian_rna"  # 需要编译生成的可执行文件路径
        cmd = f"{exe_path} {fg_path} {bg_path}"

        try:
            p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
            if p.returncode != 0:
                print(f"❌ 匈牙利算法执行失败：{p.stderr}")
                return []

            # 解析输出结果
            raw_pairs = []
            lines = p.stdout.strip().split('\n')
            for line in lines:
                if line.strip() and not line.startswith('Total'):
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        try:
                            i, j = int(parts[0]), int(parts[1])
                            raw_pairs.append((i - 1, j - 1))  # 转换为0-based索引
                        except ValueError:
                            continue

            return raw_pairs

        except subprocess.TimeoutExpired:
            print("❌ 匈牙利算法执行超时")
            return []
        except Exception as e:
            print(f"❌ 匈牙利算法执行出错：{str(e)}")
            return []


# === 初始化统计
total_TP, total_FP, total_FN = 0, 0, 0
f1_list = []
pred_dict = {}

with torch.no_grad():
    for i, (seq, emb) in enumerate(RNA_emb_dict.items()):
        # print(f'🔍 正在处理第 {i + 1} 个序列，长度={len(seq)}')

        emb_tensor = torch.from_numpy(emb) if isinstance(emb, np.ndarray) else emb
        emb_tensor = emb_tensor.unsqueeze(0).to(device)
        attention_mask = torch.zeros(emb_tensor.shape[:2], dtype=torch.bool).to(device)

        px, _, _ = model(emb_tensor, attention_mask=attention_mask)
        prob = F.softmax(px, dim=-1)[0, :, :, 1]
        prob_np = ((prob + prob.T) / 2).cpu().numpy()

        if seq in label_dict:
            _, label_matrix = label_dict[seq]
            label_matrix = label_matrix.cpu().numpy() if torch.is_tensor(label_matrix) else np.array(label_matrix)

            if prob_np.shape != label_matrix.shape:
                print(f"⚠️ 警告：预测矩阵和标签矩阵shape不匹配，跳过该序列。")
                continue
        else:
            print(f"⚠️ 没有找到标签结构: {seq}")
            label_matrix = None

        # 使用匈牙利算法
        pairs = run_hungarian_algorithm(prob_np)
        binary = np.zeros_like(prob_np, dtype=int)
        for i1, i2 in pairs:
            if 0 <= i1 < binary.shape[0] and 0 <= i2 < binary.shape[0]:
                binary[i1, i2] = binary[i2, i1] = 1

        pred_dict[seq] = binary

        # === 评估指标计算
        if label_matrix is not None:
            TP = np.sum((binary == 1) & (label_matrix == 1))
            FP = np.sum((binary == 1) & (label_matrix == 0))
            FN = np.sum((binary == 0) & (label_matrix == 1))

            total_TP += TP
            total_FP += FP
            total_FN += FN

            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            f1_list.append(f1)
            # print(f"📊 当前序列 F1: {f1:.4f} | P={precision:.4f} | R={recall:.4f}")

# === 总体指标
precision_all = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
recall_all = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
f1_all = 2 * precision_all * recall_all / (precision_all + recall_all) if (precision_all + recall_all) > 0 else 0

print("\n✅ 所有序列评估结果：")
print(f"准确率（Precision）: {precision_all:.4f}")
print(f"召回率（Recall）:    {recall_all:.4f}")
print(f"F1 分数:             {f1_all:.4f}")

# === 保存预测矩阵
with open(output_pkl_path, 'wb') as f:
    pickle.dump(pred_dict, f)

# === 绘制横向小提琴图（只包含 F1 分数）
import seaborn as sns
import pandas as pd

# 修复变量名错误
f1_array = np.array(f1_list)

df = pd.DataFrame({
    'Score': f1_array,
    'Metric': ['F1'] * len(f1_array)
})

# 设置风格
sns.set(style="whitegrid", font_scale=1.2)

# 绘制小提琴图
plt.figure(figsize=(7, 3))
ax = sns.violinplot(
    data=df,
    x='Score',
    y='Metric',
    orient='h',
    inner='box',
    linewidth=1.2,
    cut=1,
    width=0.3,
    palette='Blues'
)

# 美化图像
ax.set_title('F1 Score Distribution (Hungarian Algorithm)', fontsize=14)
ax.set_xlabel('F1 Score')
ax.set_ylabel('')
ax.set_xlim(0, 1.0)
ax.grid(axis='x', linestyle='--', alpha=0.5)

plt.tight_layout(pad=0.2)

# 保存图像
save_path = os.path.join(os.path.dirname(violin_plot_path), 'violin_Hungarian_TVAE1.svg')
plt.savefig(save_path, format='svg', transparent=True)
plt.close()

print(f"✅ 匈牙利算法 F1 小提琴图已保存至：{os.path.abspath(save_path)}")


# 生成CT文件


import torch
import torch.nn.functional as F
import pickle
import os
import numpy as np
import tempfile
import subprocess
from VAE3 import TransformerVAE

# === 参数设置 ===
model_path = '/home/meixy23/TVAE/Weight2/0model_checkpoint.pth'
input_emb_path = '/home/meixy23/TVAE/Rf1_RNA_emb_dict.pkl'
label_path = '/home/meixy23/TVAE/Rf1_RNA_pair_labels_dict.pkl'
ct_output_dir = '/home/meixy23/TVAE/predict/ct_outputs'  # 存储所有 CT 文件

os.makedirs(ct_output_dir, exist_ok=True)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# === 加载模型 ===
model = TransformerVAE(input_dim=640, hidden_dim=256, num_heads=8, num_layers=4, dropout=0.1).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# === 加载数据 ===
with open(input_emb_path, 'rb') as f:
    RNA_emb_dict = pickle.load(f)

# === 匈牙利算法===
def run_hungarian_algorithm(prob_matrix, lambda_param=4.2, eps=1e-8):
    n = prob_matrix.shape[0]
    bg_matrix = np.full_like(prob_matrix, 0.5)

    with tempfile.TemporaryDirectory() as tmpdir:
        fg_path = os.path.join(tmpdir, "foreground.mat")
        bg_path = os.path.join(tmpdir, "background.mat")
        np.savetxt(fg_path, prob_matrix, fmt="%.10f", delimiter=" ")
        np.savetxt(bg_path, bg_matrix, fmt="%.10f", delimiter=" ")

        exe_path = "/home/meixy23/TVAE/hungarian_rna"
        cmd = f"{exe_path} {fg_path} {bg_path}"

        try:
            p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
            if p.returncode != 0:
                print(f"❌ 匈牙利算法执行失败：{p.stderr}")
                return []

            raw_pairs = []
            lines = p.stdout.strip().split('\n')
            for line in lines:
                if line.strip() and not line.startswith('Total'):
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        try:
                            i, j = int(parts[0]), int(parts[1])
                            raw_pairs.append((i - 1, j - 1))
                        except ValueError:
                            continue
            return raw_pairs
        except Exception as e:
            print(f"❌ 匈牙利算法执行出错：{str(e)}")
            return []

# === CT 文件写入函数 ===
def write_ct_file(seq_id, sequence, pairs, save_dir):
    length = len(sequence)
    pair_map = {min(i, j): max(i, j) for i, j in pairs}
    ct_lines = [f"{length} {seq_id}"]

    for i in range(length):
        index = i + 1
        base = sequence[i]
        prev = i if i > 0 else 0
        next_ = i + 2 if i < length - 1 else 0
        pair = pair_map.get(i, 0) + 1 if i in pair_map else (i if i in pair_map.values() else 0)
        ct_lines.append(f"{index} {base} {prev} {next_} {pair} {index}")

    ct_path = os.path.join(save_dir, f"{seq_id}.ct")
    with open(ct_path, 'w') as f:
        f.write('\n'.join(ct_lines))
    print(f"✅ 已保存 CT 文件：{ct_path}")

# === 预测并生成 CT 文件 ===
with torch.no_grad():
    for i, (seq, emb) in enumerate(RNA_emb_dict.items()):
        sequence = seq.upper().replace('T', 'U')  # 替换T为U，构造碱基序列
        emb_tensor = torch.from_numpy(emb) if isinstance(emb, np.ndarray) else emb
        emb_tensor = emb_tensor.unsqueeze(0).to(device)
        attention_mask = torch.zeros(emb_tensor.shape[:2], dtype=torch.bool).to(device)

        px, _, _ = model(emb_tensor, attention_mask=attention_mask)
        prob = F.softmax(px, dim=-1)[0, :, :, 1]
        prob_np = ((prob + prob.T) / 2).cpu().numpy()

        pairs = run_hungarian_algorithm(prob_np)
        write_ct_file(f"RNA_{i+1}", sequence, pairs, ct_output_dir)
