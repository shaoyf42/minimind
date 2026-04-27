"""
实验03：注意力机制实验
目标：理解Q/K/V的作用、Multi-Head Attention和GQA
参考：model/model_minimind.py:139-220
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SimpleAttention(nn.Module):
    def __init__(self, hidden_size=64, num_heads=4, num_kv_heads=2):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)

    def forward(self, x):
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        n_rep = self.num_heads // self.num_kv_heads
        k = k.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).reshape(bsz, self.num_heads, seq_len, self.head_dim)
        v = v.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).reshape(bsz, self.num_heads, seq_len, self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))
        weights = F.softmax(scores.float(), dim=-1).type_as(q)
        output = torch.matmul(weights, v)
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        return self.o_proj(output), weights


def exp1_qkv_intuition():
    print("=" * 60)
    print("实验3.1：Q/K/V直觉理解")
    print("=" * 60)
    torch.manual_seed(42)
    sentence = ["我", "爱", "编程"]
    x = torch.randn(1, 3, 64)
    attn = SimpleAttention(64, num_heads=4, num_kv_heads=4)
    output, weights = attn(x)
    avg_weights = weights[0].mean(dim=0).detach()
    print("注意力权重（4头平均）:")
    for i, word in enumerate(sentence):
        attn_str = ", ".join([f"{sentence[j]}:{avg_weights[i][j]:.3f}" for j in range(3)])
        print(f"  {word} → {attn_str}")
    print("\n结论：每个词关注其他词的程度不同，这就是Attention")


def exp2_multihead_diversity():
    print("\n" + "=" * 60)
    print("实验3.2：Multi-Head多样性")
    print("=" * 60)
    torch.manual_seed(42)
    x = torch.randn(1, 3, 64)
    attn = SimpleAttention(64, num_heads=4, num_kv_heads=4)
    _, weights = attn(x)
    print("各头的注意力模式:")
    for h in range(4):
        w = weights[0, h].detach()
        print(f"  Head {h}:")
        for i, word in enumerate(["我", "爱", "编程"]):
            attn_str = ", ".join([f"{['我','爱','编程'][j]}:{w[i][j]:.3f}" for j in range(3)])
            print(f"    {word} → {attn_str}")
    print("\n结论：不同头关注不同的模式→多面性")


def exp3_gqa_efficiency():
    print("\n" + "=" * 60)
    print("实验3.3：GQA效率对比")
    print("=" * 60)
    hidden_size = 512
    seq_len = 128
    num_heads = 8
    x = torch.randn(1, seq_len, hidden_size)
    attn_mha = SimpleAttention(hidden_size, num_heads=num_heads, num_kv_heads=num_heads)
    attn_gqa = SimpleAttention(hidden_size, num_heads=num_heads, num_kv_heads=2)
    mha_kv_params = sum(p.numel() for p in [attn_mha.k_proj, attn_mha.v_proj])
    gqa_kv_params = sum(p.numel() for p in [attn_gqa.k_proj, attn_gqa.v_proj])
    print(f"MHA KV参数量: {mha_kv_params:,}")
    print(f"GQA KV参数量: {gqa_kv_params:,}")
    print(f"GQA节省: {(1 - gqa_kv_params/mha_kv_params)*100:.1f}%")
    with torch.no_grad():
        out_mha, _ = attn_mha(x)
        out_gqa, _ = attn_gqa(x)
    print(f"\nMHA输出范数: {out_mha.norm():.4f}")
    print(f"GQA输出范数: {out_gqa.norm():.4f}")
    print("\n结论：GQA大幅减少KV参数，输出质量接近MHA")


def exp4_causal_mask():
    print("\n" + "=" * 60)
    print("实验3.4：Causal Mask效果")
    print("=" * 60)
    seq_len = 5
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    print("Causal Mask (True=被遮蔽):")
    print(mask.int().numpy())
    print("\n含义：每个位置只能看到自己和之前的位置")
    print("位置0只能看[0]")
    print("位置1只能看[0,1]")
    print("位置4能看[0,1,2,3,4]")
    print("\n结论：Causal Mask确保语言模型只能看到'过去'的词")


if __name__ == "__main__":
    exp1_qkv_intuition()
    exp2_multihead_diversity()
    exp3_gqa_efficiency()
    exp4_causal_mask()
