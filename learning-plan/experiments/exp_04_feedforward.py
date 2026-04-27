"""
实验04：前馈网络实验
目标：理解扩张-压缩机制和SwiGLU激活函数
参考：model/model_minimind.py:225-238
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleFFN(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.w2(F.relu(self.w1(x)))


class SwiGLUFFN(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


def exp1_expand_compress():
    print("=" * 60)
    print("实验4.1：直接变换 vs 扩张-压缩")
    print("=" * 60)
    torch.manual_seed(42)
    dim = 64
    x = torch.randn(1, 10, dim)
    direct = nn.Linear(dim, dim)
    expand_compress = nn.Sequential(nn.Linear(dim, 256), nn.ReLU(), nn.Linear(256, dim))
    with torch.no_grad():
        out_direct = direct(x)
        out_expand = expand_compress(x)
    print(f"输入范数: {x.norm():.4f}")
    print(f"直接变换(64→64)输出范数: {out_direct.norm():.4f}")
    print(f"扩张-压缩(64→256→64)输出范数: {out_expand.norm():.4f}")
    print(f"直接变换输出方差: {out_direct.var():.6f}")
    print(f"扩张-压缩输出方差: {out_expand.var():.6f}")
    print("\n结论：扩张-压缩经过高维空间，表达能力更强")


def exp2_swilu_vs_relu():
    print("=" * 60)
    print("实验4.2：SwiGLU vs 普通FFN")
    print("=" * 60)
    torch.manual_seed(42)
    dim = 64
    hidden_dim = 256
    x = torch.randn(1, 10, dim)
    simple_ffn = SimpleFFN(dim, hidden_dim)
    swiglu_ffn = SwiGLUFFN(dim, hidden_dim)
    simple_params = sum(p.numel() for p in simple_ffn.parameters())
    swiglu_params = sum(p.numel() for p in swiglu_ffn.parameters())
    print(f"普通FFN参数量: {simple_params:,}")
    print(f"SwiGLU FFN参数量: {swiglu_params:,} (约{swiglu_params/simple_params:.1f}倍)")
    with torch.no_grad():
        out_simple = simple_ffn(x)
        out_swiglu = swiglu_ffn(x)
    print(f"\n普通FFN输出范数: {out_simple.norm():.4f}")
    print(f"SwiGLU FFN输出范数: {out_swiglu.norm():.4f}")
    x_range = torch.linspace(-3, 3, 100)
    relu_out = F.relu(x_range)
    silu_out = F.silu(x_range)
    print(f"\nReLU: 负值全部为0，梯度在负值区域为0")
    print(f"SiLU: 负值有小的非零输出，梯度处处非零→更平滑")
    print("\n结论：SwiGLU门控机制+SiLU激活，表达力更强")


def exp3_gate_mechanism():
    print("\n" + "=" * 60)
    print("实验4.3：门控机制详解")
    print("=" * 60)
    torch.manual_seed(42)
    dim = 8
    x = torch.randn(1, 1, dim)
    gate_proj = nn.Linear(dim, dim, bias=False)
    up_proj = nn.Linear(dim, dim, bias=False)
    with torch.no_grad():
        gate = gate_proj(x)
        up = up_proj(x)
        gate_activated = F.silu(gate)
        gated_output = gate_activated * up
    print(f"gate分支(原始): {gate[0, 0].detach().numpy().round(3)}")
    print(f"gate分支(SiLU后): {gate_activated[0, 0].detach().numpy().round(3)}")
    print(f"up分支(原始): {up[0, 0].detach().numpy().round(3)}")
    print(f"门控输出(gate*up): {gated_output[0, 0].detach().numpy().round(3)}")
    print("\n结论：gate分支控制up分支哪些信息通过（接近0=抑制，接近1=通过）")


def exp4_attention_vs_ffn():
    print("\n" + "=" * 60)
    print("实验4.4：Attention vs FeedForward 分工")
    print("=" * 60)
    print("Attention: 词与词交互 → 信息交换 → '开会讨论'")
    print("FeedForward: 每个词独立 → 深度处理 → '各自思考'")
    print()
    print("完整Transformer Block流程:")
    print("  输入x")
    print("  → RMSNorm → Attention(词间交互) → 残差 → '讨论后'")
    print("  → RMSNorm → FeedForward(独立处理) → 残差 → '思考后'")
    print("  → 输出")
    print()
    print("分工明确：先交流信息，再深度处理，缺一不可")


if __name__ == "__main__":
    exp1_expand_compress()
    exp2_swilu_vs_relu()
    exp3_gate_mechanism()
    exp4_attention_vs_ffn()
