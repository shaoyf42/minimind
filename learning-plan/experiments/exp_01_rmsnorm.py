"""
实验01：归一化对照实验
目标：通过对照实验理解为什么深层网络需要归一化，以及RMSNorm的优势
参考：model/model_minimind.py:95-105
"""

import torch
import torch.nn as nn
import math


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self.weight * (x.float() * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x)


def exp1_gradient_vanishing():
    print("=" * 60)
    print("实验1.1：梯度消失演示（无归一化）")
    print("=" * 60)
    torch.manual_seed(42)
    x = torch.randn(1, 512)
    print(f"初始标准差: {x.std().item():.6f}")
    for i in range(8):
        W = torch.randn(512, 512) * 0.5
        x = x @ W
        x = torch.relu(x)
        print(f"Layer {i}: std={x.std().item():.6f}")
    print("\n结论：无归一化时，标准差快速衰减→梯度消失")


def exp2_rmsnorm_stabilizes():
    print("\n" + "=" * 60)
    print("实验1.2：RMSNorm稳定训练")
    print("=" * 60)
    torch.manual_seed(42)
    x = torch.randn(1, 512)
    norm = RMSNorm(512)
    print(f"初始标准差: {x.std().item():.6f}")
    for i in range(8):
        W = torch.randn(512, 512) * 0.5
        x = norm(x)
        x = x @ W
        x = torch.relu(x)
        print(f"Layer {i}: std={x.std().item():.6f}")
    print("\n结论：RMSNorm使标准差保持稳定")


def exp3_pre_ln_vs_post_ln():
    print("\n" + "=" * 60)
    print("实验1.3：Pre-LN vs Post-LN 对比")
    print("=" * 60)

    dim = 64
    norm = RMSNorm(dim)
    W_attn = nn.Linear(dim, dim, bias=False)
    W_ffn = nn.Linear(dim, dim, bias=False)

    x_pre = torch.randn(1, 10, dim)
    x_post = torch.randn(1, 10, dim)

    for layer in range(16):
        # Pre-LN
        residual_pre = x_pre
        x_pre = x_pre + W_attn(norm(x_pre))
        x_pre = x_pre + W_ffn(norm(x_pre))

        # Post-LN
        residual_post = x_post
        x_post = norm(x_post + W_attn(x_post))
        x_post = norm(x_post + W_ffn(x_post))

        if layer % 4 == 0 or layer == 15:
            print(f"Layer {layer:2d}: Pre-LN std={x_pre.std().item():.4f} | Post-LN std={x_post.std().item():.4f}")

    print("\n结论：Pre-LN在深层网络中更稳定")


if __name__ == "__main__":
    exp1_gradient_vanishing()
    exp2_rmsnorm_stabilizes()
    exp3_pre_ln_vs_post_ln()
