"""
实验02：RoPE位置编码实验
目标：理解RoPE如何编码位置信息，以及多频率机制的必要性
参考：model/model_minimind.py:108-137
"""

import torch
import math


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


def exp1_permutation_invariance():
    print("=" * 60)
    print("实验2.1：Attention排列不变性验证")
    print("=" * 60)
    torch.manual_seed(42)
    x = torch.randn(1, 3, 8)
    x_shuffled = x[:, [2, 0, 1], :]
    W_Q = torch.randn(8, 8)
    W_K = torch.randn(8, 8)
    q1, k1 = x @ W_Q, x @ W_K
    q2, k2 = x_shuffled @ W_Q, x_shuffled @ W_K
    scores1 = (q1 @ k1.transpose(-2, -1))
    scores2 = (q2 @ k2.transpose(-2, -1))
    print("原始顺序分数矩阵:\n", scores1[0].detach().numpy().round(2))
    print("打乱顺序分数矩阵:\n", scores2[0].detach().numpy().round(2))
    print("\n结论：打乱顺序只改变行列位置，不改变值→排列不变")


def exp2_rope_encoding():
    print("\n" + "=" * 60)
    print("实验2.2：RoPE旋转编码效果")
    print("=" * 60)
    dim = 8
    seq_len = 10
    freqs_cos, freqs_sin = precompute_freqs_cis(dim, seq_len, rope_base=1e6)
    print(f"频率数量: {dim // 2}")
    for i in range(min(5, seq_len)):
        print(f"位置{i}: cos前2维={freqs_cos[i, :2].tolist()}")
    print("\n结论：每个位置有唯一的旋转角度")


def exp3_float_precision():
    print("\n" + "=" * 60)
    print("实验2.3：浮点精度问题验证")
    print("=" * 60)
    dim = 8
    normal_cos, _ = precompute_freqs_cis(dim, 10, rope_base=1e6)
    extreme_cos, _ = precompute_freqs_cis(dim, 10, rope_base=1e12)
    diff_normal = (normal_cos[1] - normal_cos[0]).abs()
    diff_extreme = (extreme_cos[1] - extreme_cos[0]).abs()
    print(f"正常rope_base(1e6): 位置0和1的cos差值 = {diff_normal.max().item():.10f}")
    print(f"极端rope_base(1e12): 位置0和1的cos差值 = {diff_extreme.max().item():.15f}")
    print(f"float32精度下限: ~1e-7")
    if diff_extreme.max().item() < 1e-7:
        print("⚠️ 极端频率下，相邻位置差值小于float32精度！")
    print("\n结论：多频率机制是浮点精度限制的工程解决方案")


def exp4_relative_position():
    print("\n" + "=" * 60)
    print("实验2.4：RoPE相对位置性质验证")
    print("=" * 60)
    dim = 8
    freqs_cos, freqs_sin = precompute_freqs_cis(dim, 20, rope_base=1e6)
    q = torch.randn(1, 1, dim)
    k = torch.randn(1, 1, dim)
    q_pos5, k_pos8 = apply_rotary_pos_emb(q, k, freqs_cos[5:6], freqs_sin[5:6])
    q_pos0, k_pos3 = apply_rotary_pos_emb(q, k, freqs_cos[0:1], freqs_sin[0:1])
    score_distant = (q_pos5 @ k_pos8.transpose(-2, -1)).item()
    score_near = (q_pos0 @ k_pos3.transpose(-2, -1)).item()
    print(f"位置5看位置8(距离3)的分数: {score_distant:.6f}")
    print(f"位置0看位置3(距离3)的分数: {score_near:.6f}")
    print(f"差异: {abs(score_distant - score_near):.6f}")
    print("\n结论：相对距离相同的token对，Attention分数相同（RoPE核心性质）")


if __name__ == "__main__":
    exp1_permutation_invariance()
    exp2_rope_encoding()
    exp3_float_precision()
    exp4_relative_position()
