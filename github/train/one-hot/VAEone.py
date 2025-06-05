# === VAEone.py：基于 one-hot 输入的 Transformer-VAE，融合相对位置偏置 ===

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

########################################
# 相对位置偏置模块
########################################
class RelativePositionBias(nn.Module):
    def __init__(self, num_buckets=64, max_distance=256, n_heads=8):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.n_heads = n_heads
        self.relative_attention_bias = nn.Embedding(num_buckets, n_heads)

    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets, max_distance):
        num_buckets //= 2
        ret = (relative_position < 0).to(relative_position) * num_buckets
        relative_position = torch.abs(relative_position)
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        val_if_large = (
            max_exact
            + (
                torch.log(relative_position.float() / max_exact + 1e-6)
                / math.log(max_distance / max_exact)
                * (num_buckets - max_exact)
            ).long()
        )
        val_if_large = torch.min(
            val_if_large, torch.full_like(val_if_large, num_buckets - 1)
        )
        ret += torch.where(is_small, relative_position, val_if_large)
        return ret

    def forward(self, relative_position):
        rp_bucket = self._relative_position_bucket(
            relative_position,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance,
        )
        rp_bias = self.relative_attention_bias(rp_bucket)  # [L, L, n_heads]
        return rp_bias.permute(2, 0, 1)  # [n_heads, L, L]


########################################
# Scaled Dot-Product Attention
########################################
class ScaledDotProductAttention(nn.Module):
    def __init__(self, scale, dropout):
        super().__init__()
        self.scale = scale
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, bias=None):
        attn = torch.matmul(q / self.scale, k.transpose(-1, -2))
        if bias is not None:
            attn += bias.unsqueeze(0)  # [1, n_heads, L, L]
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)
        return output, attn


########################################
# 多头注意力模块
########################################
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_key, n_head, dropout):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_key // n_head
        self.fc_q = nn.Linear(d_model, d_key, bias=False)
        self.fc_k = nn.Linear(d_model, d_key, bias=False)
        self.fc_v = nn.Linear(d_model, d_key, bias=False)
        self.attn = ScaledDotProductAttention(scale=self.d_k ** 0.5, dropout=dropout)
        self.fc_out = nn.Linear(d_key, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def transpose_scores(self, x):
        new_shape = x.shape[:-1] + (self.n_head, self.d_k)
        return x.view(*new_shape).transpose(-3, -2)  # [B, n_head, L, d_k]

    def forward(self, x, bias):
        q = self.transpose_scores(self.fc_q(x))
        k = self.transpose_scores(self.fc_k(x))
        v = self.transpose_scores(self.fc_v(x))
        context, attn_weight = self.attn(q, k, v, bias)
        context = context.transpose(-3, -2).contiguous().view(x.size(0), x.size(1), -1)
        output = self.fc_out(context)
        return self.dropout(output), attn_weight


########################################
# Transformer 编码器层
########################################
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, d_key, n_head, dim_feedforward, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, d_key, n_head, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x, bias):
        x = x + self.attn(self.norm1(x), bias)[0]
        x = x + self.ffn(self.norm2(x))
        return x


########################################
# Transformer 编码器堆叠
########################################
class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([TransformerEncoderLayer(**kwargs) for _ in range(num_layers)])

    def forward(self, x, bias):
        for layer in self.layers:
            x = layer(x, bias)
        return x


########################################
# 主模型：Transformer-VAE with One-hot + RPE
########################################
class TransformerVAE(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=256, num_heads=8, num_layers=4, dropout=0.1):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.input_act = nn.ReLU()

        self.rpb = RelativePositionBias(n_heads=num_heads)
        self.encoder = TransformerEncoder(
            num_layers=num_layers,
            d_model=hidden_dim,
            d_key=hidden_dim,
            n_head=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout
        )

        self.fc_mu = nn.Linear(hidden_dim, hidden_dim)
        self.fc_logvar = nn.Linear(hidden_dim, hidden_dim)

        self.decoder = TransformerEncoder(
            num_layers=num_layers,
            d_model=hidden_dim,
            d_key=hidden_dim,
            n_head=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout
        )

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, attention_mask=None):
        B, L, _ = x.size()
        pos = torch.arange(L, device=x.device)
        pos = pos.unsqueeze(0) - pos.unsqueeze(1)
        bias = self.rpb(pos)  # [n_heads, L, L]

        x = self.input_act(self.input_fc(x))
        x = self.input_norm(x)
        z_enc = self.encoder(x, bias)
        mu = self.fc_mu(z_enc)
        logvar = self.fc_logvar(z_enc)
        z = self.reparameterize(mu, logvar)
        z_dec = self.decoder(z, bias)

        q = self.q_proj(z_dec)
        k = self.k_proj(z_dec)
        score = torch.matmul(q, k.transpose(-1, -2)) / (q.shape[-1] ** 0.5)  # [B, L, L]
        px = torch.stack([-score, score], dim=-1)  # [B, L, L, 2]

        return px, mu, logvar

    def kl_loss(self, mu, logvar, attention_mask=None):
        kl = 0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return -kl
