import torch
import torch.nn.functional as F
import pickle
import os
import numpy as np
import tempfile
import subprocess
import matplotlib.pyplot as plt
import seaborn as sns
from VAE3 import TransformerVAE

# === 参数设置 ===
model_path = '/home/meixy23/TVAE/Weight4/0model_checkpoint.pth'
input_emb_path = '/home/meixy23/TVAE/predict/sample666.pkl'
output_image_dir = '/home/meixy23/TVAE/predict picture'  # 图像保存目录

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# 创建输出目录
os.makedirs(output_image_dir, exist_ok=True)

# === 加载模型 ===
model = TransformerVAE(input_dim=640, hidden_dim=256, num_heads=8, num_layers=4, dropout=0.1).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# === 加载数据 ===
with open(input_emb_path, 'rb') as f:
    RNA_emb_dict = pickle.load(f)


# === 匈牙利算法 ===
def run_hungarian_algorithm(prob_matrix):
    """
    使用匈牙利算法进行RNA二级结构预测
    """
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

        except subprocess.TimeoutExpired:
            print("❌ 匈牙利算法执行超时")
            return []
        except Exception as e:
            print(f"❌ 匈牙利算法执行出错：{str(e)}")
            return []


# === 绘制配对矩阵函数 ===
def plot_pairing_matrix(matrix, title, seq_name, save_path, is_probability=False):
    """
    绘制RNA二级结构配对矩阵
    """
    plt.figure(figsize=(10, 8))

    if is_probability:
        # 概率矩阵使用连续色彩映射
        sns.heatmap(matrix,
                    cmap='Blues',
                    square=True,
                    cbar_kws={'label': 'Pairing Probability'},
                    xticklabels=False,
                    yticklabels=False)
    else:
        # 二进制矩阵使用离散色彩映射
        sns.heatmap(matrix,
                    cmap='Blues',
                    square=True,
                    cbar_kws={'label': 'Base Pairing'},
                    xticklabels=False,
                    yticklabels=False,
                    vmin=0, vmax=1)

    plt.title(f'{title}\nSequence Length: {matrix.shape[0]}',
              fontsize=12, pad=20)
    plt.xlabel('Nucleotide Position')
    plt.ylabel('Nucleotide Position')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# === 主要处理循环 ===
print(f"🚀 开始处理 {len(RNA_emb_dict)} 条RNA序列...")

with torch.no_grad():
    for idx, (seq, emb) in enumerate(RNA_emb_dict.items()):
        print(f'🔍 正在处理第 {idx + 1}/{len(RNA_emb_dict)} 个序列，长度={len(seq)}')

        # 模型预测
        emb_tensor = torch.from_numpy(emb) if isinstance(emb, np.ndarray) else emb
        emb_tensor = emb_tensor.unsqueeze(0).to(device)
        attention_mask = torch.zeros(emb_tensor.shape[:2], dtype=torch.bool).to(device)

        px, _, _ = model(emb_tensor, attention_mask=attention_mask)
        prob = F.softmax(px, dim=-1)[0, :, :, 1]
        prob_np = ((prob + prob.T) / 2).cpu().numpy()

        # 匈牙利算法获得二进制配对矩阵
        pairs = run_hungarian_algorithm(prob_np)
        binary_matrix = np.zeros_like(prob_np, dtype=int)
        for i1, i2 in pairs:
            if 0 <= i1 < binary_matrix.shape[0] and 0 <= i2 < binary_matrix.shape[0]:
                binary_matrix[i1, i2] = binary_matrix[i2, i1] = 1

        # 生成安全的文件名
        safe_seq_name = f"seq_{idx + 1:04d}"

        # 保存概率矩阵图
        prob_save_path = os.path.join(output_image_dir, f'{safe_seq_name}_probability.png')
        plot_pairing_matrix(prob_np, 'Pairing Probability Matrix', seq, prob_save_path, is_probability=True)

        # 保存预测二进制矩阵图
        pred_save_path = os.path.join(output_image_dir, f'{safe_seq_name}_predicted.png')
        plot_pairing_matrix(binary_matrix, 'Predicted Pairing Matrix', seq, pred_save_path, is_probability=False)

        print(f'  ✅ 已保存概率图: {prob_save_path}')
        print(f'  ✅ 已保存预测图: {pred_save_path}')

print(f"\n🎉 所有图像已保存至目录: {os.path.abspath(output_image_dir)}")
print(f"📊 共处理了 {len(RNA_emb_dict)} 条序列")
print(f"📁 每条序列生成了以下图像:")
print(f"   - *_probability.png: 配对概率热力图")
print(f"   - *_predicted.png: 预测配对结构图")