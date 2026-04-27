"""
实验06：预训练实践
目标：从零训练一个小型语言模型，理解预训练的完整流程
参考：trainer/train_pretrain.py

使用方式：
1. 确保已安装依赖：pip install torch
2. 运行：python exp_06_pretrain.py
3. 此脚本使用随机数据演示训练流程，实际训练请使用项目数据集

实际训练命令（需要GPU和项目数据集）：
cd /home/shao/workspace/minimind/trainer
python train_pretrain.py --data_path ../dataset/pretrain_t2t_mini.jsonl --epochs 1 --batch_size 16 --hidden_size 512
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 复用exp_05中的模型定义（简化版）
from exp_05_transformer_block import MiniMindModel


def generate_toy_data(vocab_size=100, seq_len=32, num_samples=500):
    sentences = []
    for _ in range(num_samples):
        length = torch.randint(5, seq_len, (1,)).item()
        ids = torch.randint(1, vocab_size - 1, (length,))
        sentences.append(ids)
    return sentences


def collate_fn(sentences, seq_len, vocab_size, pad_id=0):
    batch = []
    for ids in sentences:
        if len(ids) > seq_len:
            ids = ids[:seq_len]
        padded = torch.cat([ids, torch.full((seq_len - len(ids),), pad_id, dtype=torch.long)])
        batch.append(padded)
    return torch.stack(batch)


def exp1_pretrain_loop():
    print("=" * 60)
    print("实验6.1：预训练循环演示（使用随机数据）")
    print("=" * 60)
    torch.manual_seed(42)
    vocab_size = 100
    hidden_size = 64
    seq_len = 32
    batch_size = 8
    lr = 1e-3
    epochs = 3

    model = MiniMindModel(vocab_size=vocab_size, hidden_size=hidden_size, num_layers=2, num_heads=4, num_kv_heads=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")
    print(f"词表大小: {vocab_size}")
    print(f"序列长度: {seq_len}")
    print(f"训练轮数: {epochs}")
    print()

    sentences = generate_toy_data(vocab_size, seq_len)
    for epoch in range(epochs):
        total_loss = 0
        n_batches = 0
        indices = torch.randperm(len(sentences))
        for i in range(0, len(sentences), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_sentences = [sentences[idx] for idx in batch_indices]
            input_ids = collate_fn(batch_sentences, seq_len, vocab_size)
            logits = model(input_ids)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, vocab_size), shift_labels.view(-1), ignore_index=0)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        avg_loss = total_loss / n_batches
        ppl = math.exp(min(avg_loss, 20))
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | PPL: {ppl:.2f}")

    print("\n结论：loss持续下降，模型在学习预测下一个token")
    print("实际训练请使用: python train_pretrain.py --data_path ../dataset/pretrain_t2t_mini.jsonl")


def exp2_lr_schedule():
    print("\n" + "=" * 60)
    print("实验6.2：学习率调度可视化")
    print("=" * 60)
    warmup_steps = 100
    total_steps = 1000
    max_lr = 5e-4
    min_lr = 1e-5
    lrs = []
    for step in range(total_steps):
        if step < warmup_steps:
            lr = max_lr * step / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))
        lrs.append(lr)
    print(f"初始lr: {lrs[0]:.6f}")
    print(f"峰值lr (step {warmup_steps}): {lrs[warmup_steps]:.6f}")
    print(f"最终lr: {lrs[-1]:.6f}")
    print("\n结论：warmup阶段线性增长，之后余弦衰减")


if __name__ == "__main__":
    exp1_pretrain_loop()
    exp2_lr_schedule()
