"""
实验05：Transformer Block组装
目标：将所有组件组装成完整的Transformer Block，验证数据流
参考：model/model_minimind.py:359-471
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self.weight * (x.float() * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x)


def precompute_freqs_cis(dim, end, rope_base=1e6):
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(end)
    freqs = torch.outer(t, freqs)
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin):
    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1]//2:], x[..., :x.shape[-1]//2]), dim=-1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Attention(nn.Module):
    def __init__(self, hidden_size=64, num_heads=4, num_kv_heads=2):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)

    def forward(self, x, freqs_cos, freqs_sin):
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        n_rep = self.num_heads // self.num_kv_heads
        k = k.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).reshape(bsz, self.num_heads, seq_len, self.head_dim)
        v = v.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).reshape(bsz, self.num_heads, seq_len, self.head_dim)
        q, k = apply_rotary_pos_emb(q, k, freqs_cos[:seq_len], freqs_sin[:seq_len])
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))
        weights = F.softmax(scores.float(), dim=-1).type_as(q)
        output = torch.matmul(weights, v)
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        return self.o_proj(output)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MiniMindBlock(nn.Module):
    def __init__(self, hidden_size=64, num_heads=4, num_kv_heads=2):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size)
        self.self_attn = Attention(hidden_size, num_heads, num_kv_heads)
        self.post_attention_layernorm = RMSNorm(hidden_size)
        self.mlp = FeedForward(hidden_size, hidden_size * 4)

    def forward(self, x, freqs_cos, freqs_sin):
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, freqs_cos, freqs_sin)
        x = x + residual
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = x + residual
        return x


class MiniMindModel(nn.Module):
    def __init__(self, vocab_size=100, hidden_size=64, num_layers=4, num_heads=4, num_kv_heads=2):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([MiniMindBlock(hidden_size, num_heads, num_kv_heads) for _ in range(num_layers)])
        self.norm = RMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight
        head_dim = hidden_size // num_heads
        freqs_cos, freqs_sin = precompute_freqs_cis(head_dim, 512)
        self.register_buffer("freqs_cos", freqs_cos)
        self.register_buffer("freqs_sin", freqs_sin)

    def forward(self, input_ids):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x, self.freqs_cos, self.freqs_sin)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits


def exp1_single_block():
    print("=" * 60)
    print("实验5.1：单个Transformer Block数据流")
    print("=" * 60)
    torch.manual_seed(42)
    block = MiniMindBlock(hidden_size=64, num_heads=4, num_kv_heads=2)
    head_dim = 64 // 4
    freqs_cos, freqs_sin = precompute_freqs_cis(head_dim, 10)
    x = torch.randn(1, 5, 64)
    print(f"输入形状: {x.shape}")
    print(f"输入范数: {x.norm():.4f}")
    with torch.no_grad():
        out = block(x, freqs_cos, freqs_sin)
    print(f"输出形状: {out.shape}")
    print(f"输出范数: {out.norm():.4f}")
    print(f"维度不变: {x.shape == out.shape}")
    print("\n结论：Block输入输出维度相同，但内容已融合上下文信息")


def exp2_full_model():
    print("\n" + "=" * 60)
    print("实验5.2：完整模型前向传播")
    print("=" * 60)
    torch.manual_seed(42)
    model = MiniMindModel(vocab_size=100, hidden_size=64, num_layers=4, num_heads=4, num_kv_heads=2)
    total_params = sum(p.numel() for p in model.parameters())
    embed_params = model.embed_tokens.weight.numel()
    print(f"模型总参数量: {total_params:,}")
    print(f"嵌入层参数量: {embed_params:,}")
    print(f"权重共享: embed_tokens和lm_head共享{embed_params:,}参数")
    input_ids = torch.randint(0, 100, (1, 5))
    print(f"\n输入token IDs: {input_ids.tolist()}")
    with torch.no_grad():
        logits = model(input_ids)
    print(f"输出logits形状: {logits.shape}")
    print(f"预测下一个token: {logits[0, -1].argmax().item()}")
    print("\n结论：完整模型可以正常前向传播，输出每个位置的词预测")


def exp3_residual_effect():
    print("\n" + "=" * 60)
    print("实验5.3：残差连接效果验证")
    print("=" * 60)
    torch.manual_seed(42)
    block_with_residual = MiniMindBlock(hidden_size=64, num_heads=4, num_kv_heads=2)
    x = torch.randn(1, 5, 64)
    head_dim = 64 // 4
    freqs_cos, freqs_sin = precompute_freqs_cis(head_dim, 10)
    with torch.no_grad():
        out_with = block_with_residual(x, freqs_cos, freqs_sin)
    print(f"输入范数: {x.norm():.4f}")
    print(f"有残差输出范数: {out_with.norm():.4f}")
    diff = (out_with - x).norm()
    print(f"输出与输入的差异范数: {diff:.4f}")
    print(f"差异/输入比: {diff/x.norm():.4f}")
    print("\n结论：残差连接使输出≈输入+小调整，训练更稳定")


if __name__ == "__main__":
    exp1_single_block()
    exp2_full_model()
    exp3_residual_effect()
