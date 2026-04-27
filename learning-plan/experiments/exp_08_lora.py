"""
实验08：LoRA微调实践
目标：使用LoRA在特定领域数据上微调，理解参数高效微调
参考：model/model_lora.py, trainer/train_lora.py

实际训练命令：
cd /home/shao/workspace/minimind/trainer
python train_lora.py --data_path ../dataset/lora_medical.jsonl --from_weight full_sft --lora_r 8 --lora_alpha 16
"""

import torch
import torch.nn as nn
import math


class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

    def forward(self, x):
        return (x @ self.lora_A @ self.lora_B) * self.scaling


class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.lora = LoRA(in_features, out_features, rank, alpha)

    def forward(self, x):
        return self.linear(x) + self.lora(x)


def exp1_lora_structure():
    print("=" * 60)
    print("实验8.1：LoRA低秩分解结构")
    print("=" * 60)
    in_features = 512
    out_features = 512
    rank = 8

    original_params = in_features * out_features
    lora_params = in_features * rank + rank * out_features
    print(f"原始Linear参数量: {original_params:,}")
    print(f"LoRA参数量 (rank={rank}): {lora_params:,}")
    print(f"参数比例: {lora_params/original_params*100:.2f}%")
    print(f"压缩比: {original_params/lora_params:.1f}x")
    print("\n结论：LoRA仅增加极少参数（<2%），即可实现高效微调")


def exp2_lora_initialization():
    print("\n" + "=" * 60)
    print("实验8.2：LoRA初始化验证")
    print("=" * 60)
    torch.manual_seed(42)
    in_features = 64
    out_features = 64
    rank = 4

    lora = LoRA(in_features, out_features, rank)
    print(f"A矩阵 (高斯初始化): 均值={lora.lora_A.mean():.6f}, 标准差={lora.lora_A.std():.6f}")
    print(f"B矩阵 (零初始化): 均值={lora.lora_B.mean():.6f}, 标准差={lora.lora_B.std():.6f}")
    x = torch.randn(1, in_features)
    lora_output = lora(x)
    print(f"LoRA初始输出范数: {lora_output.norm():.8f}")
    print("\n结论：B矩阵零初始化→LoRA初始输出为0→不改变原始模型行为")


def exp3_lora_training():
    print("\n" + "=" * 60)
    print("实验8.3：LoRA训练过程演示")
    print("=" * 60)
    torch.manual_seed(42)
    in_features = 64
    out_features = 64
    rank = 4

    lora_linear = LoRALinear(in_features, out_features, rank)
    for name, param in lora_linear.linear.named_parameters():
        param.requires_grad = False

    trainable = sum(p.numel() for p in lora_linear.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in lora_linear.parameters() if not p.requires_grad)
    print(f"可训练参数(LoRA): {trainable:,}")
    print(f"冻结参数(原始): {frozen:,}")
    print(f"训练比例: {trainable/(trainable+frozen)*100:.2f}%")

    x = torch.randn(4, in_features)
    target = torch.randn(4, out_features)
    optimizer = torch.optim.AdamW([p for p in lora_linear.parameters() if p.requires_grad], lr=1e-3)

    for step in range(20):
        output = lora_linear(x)
        loss = F.mse_loss(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 5 == 0:
            print(f"Step {step:3d} | Loss: {loss.item():.6f}")

    print("\n结论：仅训练LoRA参数即可适配新任务，原始参数保持不变")


if __name__ == "__main__":
    exp1_lora_structure()
    exp2_lora_initialization()
    exp3_lora_training()
